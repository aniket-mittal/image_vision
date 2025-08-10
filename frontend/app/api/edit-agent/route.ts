import { NextRequest, NextResponse } from "next/server"
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"

const MODEL_SERVER_URL = (process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "")
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || ""
const STABILITY_API_KEY = process.env.STABILITY_API_KEY || process.env.STABILITY_KEY || ""

type EditModel = "LaMa" | "SDXL" | "OpenAIImages" | "StabilityAI"

interface EditAgentInput {
  imageData: string
  instruction: string
  aiModel: EditModel
  previewOnly?: boolean
}

interface StepLog {
  step: string
  info?: any
  image?: string | null
}

async function saveTempImage(imageData: string, basename = "edit_input.jpg") {
  const tempDir = join(tmpdir(), "image_vision_edit")
  await mkdir(tempDir, { recursive: true })
  const imageBuffer = Buffer.from(imageData.split(",")[1], "base64")
  const imagePath = join(tempDir, `${Date.now()}_${basename}`)
  await writeFile(imagePath, imageBuffer)
  return { imagePath, tempDir }
}

async function callModelServer(path: string, body: any) {
  const r = await fetch(`${MODEL_SERVER_URL}/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${path} ${r.status}: ${await r.text()}`)
  return r.json()
}

async function callFaceParse(imageData: string) {
  return callModelServer("face_parse", { image_data: imageData }) as Promise<{ parsing: { width: number; height: number }; face_mask_png: string }>
}

async function callMattingRefine(imageData: string, maskPng: string) {
  return callModelServer("matting_refine", { image_data: imageData, mask_png: maskPng }) as Promise<{ refined_mask_png: string }>
}

function parseIntent(instruction: string) {
  const t = instruction.toLowerCase()
  const isLighting = /brighten|brighter|lighting|exposure|contrast|tone|color balance|white balance|denoise/.test(t)
  const isRemove = /(remove|erase|get rid of)\s+([\w\s-]+)|remove the|erase the/.test(t)
  const isReplace = /(replace|swap)\s+([\w\s-]+)\s+with\s+([\w\s-]+)/.test(t)
  let target = "object"
  let replacement: string | null = null
  // naive target extraction
  const mRemove = t.match(/(?:remove|erase|get rid of)\s+([^.,;]+)/)
  if (mRemove) target = mRemove[1].trim()
  const mReplace = t.match(/(?:replace|swap)\s+([^.,;]+?)\s+with\s+([^.,;]+)/)
  if (mReplace) { target = mReplace[1].trim(); replacement = mReplace[2].trim() }
  const mentionsFace = /(face|skin|person|people|human|portrait)/.test(t)
  return { isLighting, isRemove: !!mRemove, isReplace: !!mReplace, target, replacement, mentionsFace }
}

async function buildMask(imagePath: string, imageData: string, prompt: string, opts?: { dilate?: number; feather?: number }) {
  const body = {
    image_data: imageData,
    prompt,
    dilate_px: opts?.dilate ?? 3,
    feather_px: opts?.feather ?? 3,
    return_rgba: true,
  }
  return callModelServer("mask_from_text", body) as Promise<{ mask_png: string; mask_binary_shape: number[]; mask_binary_sum: number }>
}

async function enhanceLocal(imagePath: string, imageData: string, params?: any) {
  const body = {
    image_data: imageData,
    white_balance: params?.white_balance ?? true,
    clahe: params?.clahe ?? true,
    gamma: params?.gamma ?? 0.9,
    unsharp_amount: params?.unsharp_amount ?? 0.5,
    unsharp_radius: params?.unsharp_radius ?? 3,
    denoise: params?.denoise ?? false,
  }
  return callModelServer("enhance_local", body) as Promise<{ processedImageData: string }>
}

async function inpaintPreviewLaMa(imageData: string, maskPng: string) {
  return callModelServer("inpaint_lama", { image_data: imageData, mask_png: maskPng }) as Promise<{ processedImageData: string }>
}

async function inpaintSDXL(imageData: string, maskPng: string, prompt: string, params?: any) {
  const body = {
    image_data: imageData,
    mask_png: maskPng,
    prompt,
    negative_prompt: params?.negative_prompt || "low quality, blurry, artifacts",
    guidance_scale: params?.guidance_scale ?? 6.0,
    num_inference_steps: params?.num_inference_steps ?? 30,
    use_canny: params?.use_canny ?? true,
    use_depth: params?.use_depth ?? false,
    seed: params?.seed ?? 0,
  }
  return callModelServer("inpaint_sdxl", body) as Promise<{ processedImageData: string }>
}

async function openAIImagesEdit(imageData: string, maskPng: string, prompt: string) {
  // https://api.openai.com/v1/images/edits (multipart)
  if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY missing")
  const form = new FormData()
  // Convert data URLs to Blobs
  const imageBuf = Buffer.from(imageData.split(",")[1], "base64")
  const maskBuf = Buffer.from(maskPng.split(",")[1], "base64")
  form.append("image", new Blob([imageBuf], { type: "image/png" }), "image.png")
  form.append("mask", new Blob([maskBuf], { type: "image/png" }), "mask.png")
  form.append("prompt", prompt)
  form.append("size", "1024x1024")
  const r = await fetch("https://api.openai.com/v1/images/edits", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: form as any,
  })
  if (!r.ok) throw new Error(`openai images ${r.status}: ${await r.text()}`)
  const data = await r.json()
  const b64 = data.data?.[0]?.b64_json
  if (!b64) throw new Error("openai images: empty result")
  return { processedImageData: `data:image/png;base64,${b64}` }
}

async function stabilityInpaint(imageData: string, maskPng: string, prompt: string) {
  if (!STABILITY_API_KEY) throw new Error("STABILITY_API_KEY missing")
  // v2beta inpaint
  const form = new FormData()
  const imageBuf = Buffer.from(imageData.split(",")[1], "base64")
  const maskBuf = Buffer.from(maskPng.split(",")[1], "base64")
  form.append("prompt", prompt)
  form.append("image", new Blob([imageBuf], { type: "image/png" }), "image.png")
  form.append("mask", new Blob([maskBuf], { type: "image/png" }), "mask.png")
  const r = await fetch("https://api.stability.ai/v2beta/stable-image/edit/inpaint", {
    method: "POST",
    headers: { Authorization: `Bearer ${STABILITY_API_KEY}` },
    body: form as any,
  })
  if (!r.ok) throw new Error(`stability ${r.status}: ${await r.text()}`)
  const contentType = r.headers.get("content-type") || ""
  if (contentType.includes("application/json")) {
    const j = await r.json()
    const b64 = j?.artifacts?.[0]?.base64
    if (!b64) throw new Error("stability: empty result")
    return { processedImageData: `data:image/png;base64,${b64}` }
  } else {
    const arr = new Uint8Array(await r.arrayBuffer())
    const b64 = Buffer.from(arr).toString("base64")
    return { processedImageData: `data:image/png;base64,${b64}` }
  }
}

export async function POST(req: NextRequest) {
  try {
    const { imageData, instruction, aiModel, previewOnly } = (await req.json()) as EditAgentInput
    if (!imageData || !instruction || !aiModel) {
      return NextResponse.json({ error: "Missing imageData, instruction, or aiModel" }, { status: 400 })
    }

    const steps: StepLog[] = []
    const { imagePath } = await saveTempImage(imageData)
    steps.push({ step: "upload", info: { imagePath } })

    // Analyze intent and targets
    const intent = parseIntent(instruction)
    steps.push({ step: "analyze", info: intent })

    // If lighting/tone -> local enhance
    if (intent.isLighting && !intent.isRemove && !intent.isReplace) {
      // If face-related, get face mask and apply targeted enhancement
      let maskPng: string | undefined = undefined
      if (intent.mentionsFace) {
        try {
          const fp = await callFaceParse(imageData)
          maskPng = fp.face_mask_png
          steps.push({ step: "face_parse", image: maskPng })
        } catch {}
      }
      const enh = await enhanceLocal(imagePath, imageData, { gamma: 0.9, mask_png: maskPng })
      steps.push({ step: "enhance_local", image: enh.processedImageData })
      return NextResponse.json({ ok: true, result: enh.processedImageData, steps })
    }

    // Else we need a mask
    const mask = await buildMask(imagePath, imageData, intent.target || "object", { dilate: 3, feather: 3 })
    if (!mask.mask_png) {
      return NextResponse.json({ error: "Failed to build mask" }, { status: 502 })
    }
    steps.push({ step: "mask", info: { pixels: mask.mask_binary_sum }, image: mask.mask_png })
    // Try matting refine to improve edges
    let refinedMask = mask.mask_png
    try {
      const r = await callMattingRefine(imageData, mask.mask_png)
      if (r?.refined_mask_png) {
        refinedMask = r.refined_mask_png
        steps.push({ step: "matting_refine", image: refinedMask })
      }
    } catch {}

    // Routing and execution
    let edited: { processedImageData: string } | null = null
    if (aiModel === "LaMa" || previewOnly) {
      edited = await inpaintPreviewLaMa(imageData, refinedMask)
      steps.push({ step: "inpaint_lama", image: edited.processedImageData })
      // Optional validate
      const val = await callModelServer("validate_edit", {
        original_image_data: imageData,
        edited_image_data: edited.processedImageData,
        mask_png: refinedMask,
        concept: intent.target || "object",
      })
      steps.push({ step: "validate", info: val })
      return NextResponse.json({ ok: true, result: edited.processedImageData, steps })
    }

    if (aiModel === "SDXL") {
      const prompt = intent.isReplace && intent.replacement
        ? `Replace ${intent.target} with ${intent.replacement}`
        : `Remove ${intent.target}`
      let params: any = { guidance_scale: 6.0, num_inference_steps: 30, use_canny: true, use_depth: false }
      let attempt = 0
      let best = null as any
      while (attempt < 2) {
        const out = await inpaintSDXL(imageData, refinedMask, prompt, params)
        const val = await callModelServer("validate_edit", {
          original_image_data: imageData,
          edited_image_data: out.processedImageData,
          mask_png: refinedMask,
          concept: intent.target || "object",
        })
        steps.push({ step: `inpaint_sdxl_${attempt+1}`, image: out.processedImageData })
        steps.push({ step: `validate_${attempt+1}`, info: val })
        best = { out, val }
        if (val.passed) { edited = out; break }
        // Retry: widen mask feather and adjust params
        attempt += 1
        params = { ...params, guidance_scale: params.guidance_scale + 0.8, num_inference_steps: params.num_inference_steps + 10 }
      }
      if (!edited && best) edited = best.out
      return NextResponse.json({ ok: true, result: edited?.processedImageData || null, steps })
    }

    if (aiModel === "OpenAIImages") {
      const prompt = intent.isReplace && intent.replacement
        ? `Replace ${intent.target} with ${intent.replacement}`
        : `Remove ${intent.target}`
      const out = await openAIImagesEdit(imageData, refinedMask, prompt)
      steps.push({ step: "openai_images_edit", image: out.processedImageData })
      const val = await callModelServer("validate_edit", {
        original_image_data: imageData,
        edited_image_data: out.processedImageData,
        mask_png: refinedMask,
        concept: intent.target || "object",
      })
      steps.push({ step: "validate", info: val })
      return NextResponse.json({ ok: true, result: out.processedImageData, steps })
    }

    if (aiModel === "StabilityAI") {
      const prompt = intent.isReplace && intent.replacement
        ? `Replace ${intent.target} with ${intent.replacement}`
        : `Remove ${intent.target}`
      const out = await stabilityInpaint(imageData, refinedMask, prompt)
      steps.push({ step: "stability_inpaint", image: out.processedImageData })
      const val = await callModelServer("validate_edit", {
        original_image_data: imageData,
        edited_image_data: out.processedImageData,
        mask_png: refinedMask,
        concept: intent.target || "object",
      })
      steps.push({ step: "validate", info: val })
      return NextResponse.json({ ok: true, result: out.processedImageData, steps })
    }

    return NextResponse.json({ error: "Unsupported aiModel" }, { status: 400 })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || "error" }, { status: 500 })
  }
}


