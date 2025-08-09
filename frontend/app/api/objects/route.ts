import { NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { mkdir, writeFile, readFile } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"

// In-memory cache for detected objects per image hash
// Key: simple hash of image base64; Value: detection result JSON
const objectCache = new Map<string, any>()

function simpleHash(input: string): string {
  let hash = 0
  for (let i = 0; i < input.length; i++) {
    const chr = input.charCodeAt(i)
    hash = (hash << 5) - hash + chr
    hash |= 0
  }
  return hash.toString()
}

async function detectAllObjects(imagePath: string, prompt = "object"): Promise<any> {
  // Prefer OBJECTS_SERVER_URL if provided, else use MODEL_SERVER_URL if available
  const OBJECTS_SERVER_URL = process.env.OBJECTS_SERVER_URL
  const MODEL_SERVER_URL = process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765"
  const baseUrl = OBJECTS_SERVER_URL || MODEL_SERVER_URL
  try {
    const resp = await fetch(`${baseUrl}/detect_all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_path: imagePath, prompt }),
    })
    if (resp.ok) return resp.json()
  } catch (_) {
    // fall back to local
  }
  // Fallback: local spawn
  return new Promise((resolve, reject) => {
    const cwd = "/Users/aniketmittal/Desktop/code/image_vision"
    const py = spawn("conda", ["run", "-n", "clip_api", "python", join(cwd, "run_detect_all.py"), imagePath, "--prompt", prompt], { cwd })

    let out = ""
    let err = ""
    py.stdout.on("data", (d) => (out += d.toString()))
    py.stderr.on("data", (d) => (err += d.toString()))
    py.on("close", (code) => {
      if (code === 0) {
        try {
          const json = JSON.parse(out.trim())
          resolve(json)
        } catch (e) {
          reject(new Error(`Failed parsing detector JSON: ${e}`))
        }
      } else {
        reject(new Error(`Detector exit ${code}: ${err}`))
      }
    })
  })
}

export async function POST(req: NextRequest) {
  try {
    const { imageData, prompt } = await req.json()
    if (!imageData) {
      return NextResponse.json({ error: "Missing imageData" }, { status: 400 })
    }

    // Compute cache key and return if exists
    const key = simpleHash(imageData)
    if (objectCache.has(key)) {
      return NextResponse.json({ cached: true, ...objectCache.get(key) })
    }

    const base64 = imageData.replace(/^data:image\/[a-zA-Z]+;base64,/, "")
    const buf = Buffer.from(base64, "base64")
    const tempDir = join(tmpdir(), "image_vision_objects")
    await mkdir(tempDir, { recursive: true })
    const imagePath = join(tempDir, `upload_${Date.now()}.jpg`)
    await writeFile(imagePath, buf)

    const result = await detectAllObjects(imagePath, prompt || "object")
    objectCache.set(key, result)

    return NextResponse.json({ cached: false, ...result })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || "Detection failed" }, { status: 500 })
  }
}

export async function PUT(req: NextRequest) {
  // SAM click inference: expects { imageData, x, y, blurStrength }
  try {
    const { imageData, x, y, blurStrength } = await req.json()
    if (!imageData || typeof x !== "number" || typeof y !== "number") {
      return NextResponse.json({ error: "Missing imageData or x/y" }, { status: 400 })
    }
    const base64 = imageData.replace(/^data:image\/[a-zA-Z]+;base64,/, "")
    const buf = Buffer.from(base64, "base64")
    const tempDir = join(tmpdir(), "image_vision_objects")
    await mkdir(tempDir, { recursive: true })
    const imagePath = join(tempDir, `click_${Date.now()}.jpg`)
    await writeFile(imagePath, buf)

    const OBJECTS_SERVER_URL = process.env.OBJECTS_SERVER_URL
    const MODEL_SERVER_URL = process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765"
    const baseUrl = OBJECTS_SERVER_URL || MODEL_SERVER_URL
    if (baseUrl) {
      const resp = await fetch(`${baseUrl}/sam_click`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_path: imagePath, x, y, blur_strength: blurStrength ?? 15 }),
      })
      if (!resp.ok) return NextResponse.json({ error: `SAM click failed ${resp.status}` }, { status: 500 })
      const payload = await resp.json()
      // model server returns processedImageData directly
      if (payload.processedImageData) {
        return NextResponse.json({ processedImageData: payload.processedImageData, bbox: payload.bbox })
      }
      // backward compat if only path returned
      const abs = payload.saved
      const data = await import("fs/promises").then((fs) => fs.readFile(abs))
      const processedImageData = `data:image/jpeg;base64,${data.toString("base64")}`
      return NextResponse.json({ processedImageData, bbox: payload.bbox })
    }

    // Fallback: local spawn
    const cwd = "/Users/aniketmittal/Desktop/code/image_vision"
    const args = [
      "run_sam_click.py",
      imagePath,
      String(Math.round(x)),
      String(Math.round(y)),
      "--blur_strength",
      String(blurStrength ?? 15),
      "--output_dir",
      "temp_output",
    ]
    const py = spawn("conda", ["run", "-n", "clip_api", "python", ...args], { cwd })
    let out = ""
    let err = ""
    py.stdout.on("data", (d) => (out += d.toString()))
    py.stderr.on("data", (d) => (err += d.toString()))
    const done: Promise<Response> = new Promise((resolve) => {
      py.on("close", async (code) => {
        if (code !== 0) {
          resolve(NextResponse.json({ error: `SAM click failed: ${err}` }, { status: 500 }))
          return
        }
        try {
          const parsed = JSON.parse(out.trim())
          const abs = join(cwd, parsed.saved)
          const data = await import("fs/promises").then((fs) => fs.readFile(abs))
          const processedImageData = `data:image/jpeg;base64,${data.toString("base64")}`
          resolve(NextResponse.json({ processedImageData, bbox: parsed.bbox }))
        } catch (e: any) {
          resolve(NextResponse.json({ error: `Parse/read failed: ${e?.message}` }, { status: 500 }))
        }
      })
    })
    return done
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || "Click inference failed" }, { status: 500 })
  }
}


