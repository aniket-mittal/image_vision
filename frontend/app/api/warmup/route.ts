import { NextRequest, NextResponse } from "next/server"
import { spawn, type ChildProcessWithoutNullStreams } from "child_process"
import { mkdir, writeFile } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"

let warmedUp = false
let modelServer: ChildProcessWithoutNullStreams | null = null
let modelServerStarting = false

async function startModelServer(): Promise<void> {
  if (modelServer || modelServerStarting) return
  modelServerStarting = true
  const cwd = "/Users/aniketmittal/Desktop/code/image_vision"
  console.log("[Warmup] Starting CLIP model server...")
  modelServer = spawn(
    "conda",
    ["run", "-n", "clip_api", "python", "model_server.py"],
    { cwd, detached: true }
  )
  modelServer.unref()
  // Wait until health endpoint responds
  const maxAttempts = 20
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const res = await fetch("http://127.0.0.1:8765/health")
      if (res.ok) {
        console.log("[Warmup] CLIP model server is healthy")
        modelServerStarting = false
        return
      }
    } catch (e) {
      // ignore
    }
    await new Promise((r) => setTimeout(r, 300))
  }
  console.warn("[Warmup] CLIP model server health check timed out; continuing")
  modelServerStarting = false
}

async function createTinyImage(): Promise<string> {
  const tempDir = join(tmpdir(), "image_vision_warmup")
  await mkdir(tempDir, { recursive: true })
  const p = join(tempDir, `tiny_${Date.now()}.png`)
  // 1x1 transparent PNG
  const tinyPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAuMB9YzqY5kAAAAASUVORK5CYII="
  await writeFile(p, Buffer.from(tinyPngBase64, "base64"))
  return p
}

function runPy(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const cwd = "/Users/aniketmittal/Desktop/code/image_vision"
    // Use conda env if available
    const py = spawn("conda", ["run", "-n", "clip_api", "python", ...args], { cwd })
    let err = ""
    py.stderr.on("data", (d) => (err += d.toString()))
    py.on("close", (code) => {
      if (code === 0) resolve()
      else reject(new Error(`Warmup failed (${args.join(" ")}) ${code}: ${err}`))
    })
  })
}

export async function POST(_req: NextRequest) {
  if (warmedUp) {
    console.log("[Warmup] Already warmed up")
    return NextResponse.json({ status: "ok", warmedUp })
  }
  try {
    const img = await createTinyImage()
    console.log("[Warmup] Starting CLIP warmup...")
    // Preload CLIP attention
    await runPy(["run_attention_masks.py", img, "warmup", "--output_dir", "temp_output"]) 
    console.log("[Warmup] CLIP warmup done")
    await startModelServer()
    // Preload Grounded SAM by running detection (and mask pipeline) once
    console.log("[Warmup] Starting Grounded-SAM warmup...")
    await runPy(["run_detect_all.py", img, "--prompt", "object"]) 
    console.log("[Warmup] Grounded-SAM warmup done")
    warmedUp = true
    return NextResponse.json({ status: "ok", warmedUp: true })
  } catch (e: any) {
    // Do not fail the app on warmup; just report error
    return NextResponse.json({ status: "error", message: e?.message || "warmup failed" }, { status: 200 })
  }
}


