import { NextRequest, NextResponse } from "next/server"
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

async function detectAllObjects(imagePath: string, prompt = "object", imageBase64?: string): Promise<any> {
  // Prefer OBJECTS_SERVER_URL if provided, else use MODEL_SERVER_URL if available
  const OBJECTS_SERVER_URL = (process.env.OBJECTS_SERVER_URL || "").replace(/\/+$/, "")
  const MODEL_SERVER_URL = (process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "")
  const baseUrl = OBJECTS_SERVER_URL || MODEL_SERVER_URL
    const b64 = imageBase64 ?? ""
    const resp = await fetch(`${baseUrl}/detect_all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
      // send only image_data to support remote servers that cannot read local temp paths
      body: JSON.stringify({ image_data: b64 ? `data:image/jpeg;base64,${b64}` : undefined, prompt }),
  })
  if (!resp.ok) throw new Error(`detect_all failed ${resp.status}`)
  return resp.json()
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

    const result = await detectAllObjects(imagePath, prompt || "object", buf.toString("base64"))
    objectCache.set(key, result)

    return NextResponse.json({ cached: false, ...result })
  } catch (e: any) {
    // Return a friendly empty response instead of 500 so the UI can continue
    return NextResponse.json({ image: null, objects: [], error: e?.message || "Detection failed" }, { status: 200 })
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

    const OBJECTS_SERVER_URL = (process.env.OBJECTS_SERVER_URL || "").replace(/\/+$/, "")
    const MODEL_SERVER_URL = (process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "")
    const baseUrl = OBJECTS_SERVER_URL || MODEL_SERVER_URL
    const b64click = buf.toString("base64")
    const resp = await fetch(`${baseUrl}/sam_click`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data: `data:image/jpeg;base64,${b64click}`, x, y, blur_strength: blurStrength ?? 15 }),
    })
    if (!resp.ok) return NextResponse.json({ error: `SAM click failed ${resp.status}` }, { status: 500 })
    const payload = await resp.json()
    if (payload.processedImageData) {
      return NextResponse.json({ processedImageData: payload.processedImageData, bbox: payload.bbox })
    }
    // backward compat if only path returned
    const abs = payload.saved
    const data = await import("fs/promises").then((fs) => fs.readFile(abs))
    const processedImageData = `data:image/jpeg;base64,${data.toString("base64")}`
    return NextResponse.json({ processedImageData, bbox: payload.bbox })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || "Click inference failed" }, { status: 500 })
  }
}


