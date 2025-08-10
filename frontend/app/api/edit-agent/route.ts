import { NextRequest, NextResponse } from "next/server"
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"
import { createCanvas, loadImage } from "canvas"

const MODEL_SERVER_URL = (process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "")
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || ""
const STABILITY_API_KEY = process.env.STABILITY_API_KEY || process.env.STABILITY_KEY || ""

type EditModel = "SDXL" | "OpenAIImages" | "StabilityAI"

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
  console.log(`[edit-agent] Calling model server: ${path}`)
  try {
    const r = await fetch(`${MODEL_SERVER_URL}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      // Add timeout to prevent hanging requests
      signal: AbortSignal.timeout(30000), // 30 second timeout
    })
    if (!r.ok) {
      const errorText = await r.text()
      console.error(`[edit-agent] Model server error ${path}: ${r.status} - ${errorText}`)
      
      // Handle specific error codes
      if (r.status === 502) {
        throw new Error(`Model server ${path} endpoint is currently unavailable (502 Bad Gateway). This may be a temporary issue with the remote server.`)
      } else if (r.status === 503) {
        throw new Error(`Model server ${path} endpoint is temporarily unavailable (503 Service Unavailable). Please try again later.`)
      } else if (r.status >= 500) {
        throw new Error(`Model server ${path} endpoint server error (${r.status}): ${errorText}`)
      } else if (r.status === 400) {
        throw new Error(`Model server ${path} endpoint bad request (400): ${errorText}`)
      } else if (r.status === 404) {
        throw new Error(`Model server ${path} endpoint not found (404): ${errorText}`)
      }
      
      throw new Error(`${path} ${r.status}: ${errorText}`)
    }
    const result = await r.json()
    console.log(`[edit-agent] Model server success: ${path}`)
    return result
  } catch (error) {
    console.error(`[edit-agent] Model server call failed ${path}:`, error)
    throw error
  }
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

// LaMa removed to save VRAM; use SDXL or fallback inpainting instead

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
  if (!OPENAI_API_KEY) {
    console.error("[edit-agent] OPENAI_API_KEY is missing or empty")
    throw new Error("OPENAI_API_KEY missing - please check your environment variables")
  }
  
  // Validate API key format (should start with 'sk-')
  if (!OPENAI_API_KEY.startsWith('sk-')) {
    console.error("[edit-agent] OPENAI_API_KEY format appears invalid (should start with 'sk-')")
    console.error("[edit-agent] This might be a false positive - your key may still work")
    // Don't throw here, let the API call determine if the key is valid
  }
  
  console.log(`[edit-agent] Using OpenAI API key: ${OPENAI_API_KEY.substring(0, 7)}...`)
  
  try {
    // Preprocess both image and mask to meet OpenAI requirements
    console.log("[edit-agent] Preprocessing image and mask for OpenAI")
    const processedImage = await preprocessImageForOpenAI(imageData)
    const processedMask = await preprocessImageForOpenAI(maskPng)
    
    // Extract base64 data from processed images
    const imageBuf = Buffer.from(processedImage.split(",")[1], "base64")
    const maskBuf = Buffer.from(processedMask.split(",")[1], "base64")
    
    console.log(`[edit-agent] Processed image size: ${imageBuf.length} bytes, mask size: ${maskBuf.length} bytes`)
    
    // Validate mask format - OpenAI expects a black and white mask where white areas are edited
    console.log("[edit-agent] Validating mask format...")
    try {
      const maskImg = await loadImage(maskBuf)
      console.log(`[edit-agent] Mask dimensions: ${maskImg.width}x${maskImg.height}`)
      
      // Check if mask is roughly the right format by sampling some pixels
      const tempCanvas = createCanvas(maskImg.width, maskImg.height)
      const tempCtx = tempCanvas.getContext('2d')
      tempCtx.drawImage(maskImg, 0, 0)
      const imageData = tempCtx.getImageData(0, 0, maskImg.width, maskImg.height)
      const pixels = imageData.data
      
      let whitePixels = 0
      let blackPixels = 0
      let otherPixels = 0
      
      for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i]
        const g = pixels[i + 1]
        const b = pixels[i + 2]
        if (r > 200 && g > 200 && b > 200) whitePixels++
        else if (r < 50 && g < 50 && b < 50) blackPixels++
        else otherPixels++
      }
      
      console.log(`[edit-agent] Mask pixel analysis: White (edit areas): ${whitePixels}, Black (preserve): ${blackPixels}, Other: ${otherPixels}`)
      
      if (whitePixels === 0) {
        console.warn("[edit-agent] Warning: Mask appears to have no white areas (no areas to edit)")
      }
      if (blackPixels === 0) {
        console.warn("[edit-agent] Warning: Mask appears to have no black areas (no areas to preserve)")
      }
      
      // Ensure mask dimensions match image dimensions
      const imageImg = await loadImage(imageBuf)
      console.log(`[edit-agent] Image dimensions: ${imageImg.width}x${imageImg.height}`)
      
      if (maskImg.width !== imageImg.width || maskImg.height !== imageImg.height) {
        console.warn(`[edit-agent] Warning: Mask dimensions (${maskImg.width}x${maskImg.height}) don't match image dimensions (${imageImg.width}x${imageImg.height})`)
      }
      
    } catch (maskError) {
      console.warn("[edit-agent] Could not analyze mask format:", maskError)
    }
    
    const form = new FormData()
    form.append("image", new Blob([imageBuf], { type: "image/png" }), "image.png")
    form.append("mask", new Blob([maskBuf], { type: "image/png" }), "mask.png")
    form.append("prompt", prompt)
    form.append("size", "1024x1024")
    
    console.log("[edit-agent] Sending request to OpenAI with:")
    console.log(`[edit-agent] - Image size: ${imageBuf.length} bytes`)
    console.log(`[edit-agent] - Mask size: ${maskBuf.length} bytes`)
    console.log(`[edit-agent] - Prompt: "${prompt}"`)
    console.log(`[edit-agent] - Size: 1024x1024`)
    console.log(`[edit-agent] - Form data entries: ${Array.from(form.entries()).map(([k, v]) => `${k}: ${v instanceof Blob ? `${v.type} (${v.size} bytes)` : v}`).join(', ')}`)
  
    const r = await fetch("https://api.openai.com/v1/images/edits", {
      method: "POST",
      headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
      body: form as any,
    })
  
    if (!r.ok) {
      const errorText = await r.text()
      console.error(`[edit-agent] OpenAI API error ${r.status}:`, errorText)
      
      // Handle common OpenAI API errors
      if (r.status === 400) {
        try {
          const errorData = JSON.parse(errorText)
          if (errorData.error?.message?.includes("image")) {
            throw new Error(`OpenAI API: Image format/size issue - ${errorData.error.message}`)
          } else if (errorData.error?.message?.includes("mask")) {
            throw new Error(`OpenAI API: Mask format issue - ${errorData.error.message}`)
          } else if (errorData.error?.message?.includes("prompt")) {
            throw new Error(`OpenAI API: Prompt issue - ${errorData.error.message}`)
          }
        } catch (parseError) {
          // If we can't parse the error, just use the raw text
        }
        throw new Error(`OpenAI API error (400): ${errorText}`)
      } else if (r.status === 401) {
        throw new Error("OpenAI API: Authentication failed - check your API key")
      } else if (r.status === 429) {
        throw new Error("OpenAI API: Rate limit exceeded - try again later")
      } else if (r.status >= 500) {
        throw new Error(`OpenAI API server error (${r.status}): ${errorText}`)
      }
      
      throw new Error(`OpenAI API error ${r.status}: ${errorText}`)
    }
  
    const data = await r.json()
    console.log("[edit-agent] OpenAI API response:", JSON.stringify(data, null, 2))
    
    // Check for different response formats
    let b64 = data.data?.[0]?.b64_json
    if (!b64) {
      // Try alternative response format - OpenAI might return a URL instead of base64
      b64 = data.data?.[0]?.url
      if (b64) {
        console.log("[edit-agent] Found URL in response, converting to base64")
        try {
          // Download the image and convert to base64
          const imageResponse = await fetch(b64)
          if (!imageResponse.ok) {
            throw new Error(`Failed to download image from URL: ${imageResponse.status}`)
          }
          const imageBuffer = await imageResponse.arrayBuffer()
          b64 = Buffer.from(imageBuffer).toString('base64')
          console.log("[edit-agent] Successfully converted URL to base64")
        } catch (downloadError) {
          console.error("[edit-agent] Failed to download image from URL:", downloadError)
          throw new Error(`Failed to download image from OpenAI URL: ${downloadError}`)
        }
      }
    }
    
    if (!b64) {
      console.error("[edit-agent] No image data found in response. Response structure:", data)
      console.error("[edit-agent] Expected fields: data[0].b64_json or data[0].url")
      throw new Error(`No image data in response. Response keys: ${Object.keys(data)}, data keys: ${data.data ? Object.keys(data.data[0] || {}) : 'no data'}`)
    }
    
    return { processedImageData: `data:image/png;base64,${b64}` }
  } catch (error) {
    console.error("[edit-agent] OpenAI image edit failed:", error)
    throw error
  }
}

async function stabilityInpaint(imageData: string, maskPng: string, prompt: string) {
  if (!STABILITY_API_KEY) throw new Error("STABILITY_API_KEY missing")
  
  try {
    // Preprocess both image and mask for consistency
    console.log("[edit-agent] Preprocessing image and mask for Stability AI")
    const processedImage = await preprocessImageForOpenAI(imageData)
    const processedMask = await preprocessImageForOpenAI(maskPng)
    
    // Extract base64 data from processed images
    const imageBuf = Buffer.from(processedImage.split(",")[1], "base64")
    const maskBuf = Buffer.from(processedMask.split(",")[1], "base64")
    
    console.log(`[edit-agent] Processed image size: ${imageBuf.length} bytes, mask size: ${maskBuf.length} bytes`)
    
    const form = new FormData()
    form.append("prompt", prompt)
    form.append("image", new Blob([imageBuf], { type: "image/png" }), "image.png")
    form.append("mask", new Blob([maskBuf], { type: "image/png" }), "mask.png")
    
    const r = await fetch("https://api.stability.ai/v2beta/stable-image/edit/inpaint", {
      method: "POST",
      headers: { 
        Authorization: `Bearer ${STABILITY_API_KEY}`,
        Accept: "application/json"
      },
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
  } catch (error) {
    console.error("[edit-agent] Stability AI inpainting failed:", error)
    throw error
  }
}

/**
 * Validate model compatibility with the given task
 */
function validateModelForTask(aiModel: EditModel, intent: any): boolean {
  // OpenAI Images is best for object removal/replacement
  if (aiModel === "OpenAIImages") {
    return intent.isRemove || intent.isReplace
  }
  
  // SDXL is good for all tasks, especially complex edits
  if (aiModel === "SDXL") {
    return true
  }
  
  // Stability AI is good for inpainting and style changes
  if (aiModel === "StabilityAI") {
    return intent.isRemove || intent.isReplace || intent.isLighting
  }
  
  return false
}

/**
 * Preprocess image to meet OpenAI's requirements:
 * - Convert to PNG format
 * - Resize if too large (max 4MB)
 * - Ensure proper dimensions
 */
async function preprocessImageForOpenAI(imageData: string, maxSizeBytes: number = 4 * 1024 * 1024): Promise<string> {
  try {
    console.log(`[edit-agent] Preprocessing image: original size ${imageData.length} chars`)
    
    // Remove data URL prefix to get base64 data
    const base64Data = imageData.replace(/^data:image\/[a-z]+;base64,/, '')
    const imageBuffer = Buffer.from(base64Data, 'base64')
    
    console.log(`[edit-agent] Image buffer size: ${imageBuffer.length} bytes`)
    
    // Check if image is already small enough
    if (imageBuffer.length <= maxSizeBytes) {
      // If it's already PNG and small enough, return as is
      if (imageData.includes('image/png')) {
        console.log(`[edit-agent] Image already PNG and under size limit, returning as-is`)
        return imageData
      }
    }
    
    // Load the image using canvas
    const img = await loadImage(imageBuffer)
    
    // Calculate target dimensions to reduce file size
    let targetWidth = img.width
    let targetHeight = img.height
    
    // If image is very large, scale it down proportionally
    if (img.width > 1024 || img.height > 1024) {
      const aspectRatio = img.width / img.height
      if (aspectRatio > 1) {
        targetWidth = 1024
        targetHeight = Math.round(1024 / aspectRatio)
      } else {
        targetHeight = 1024
        targetWidth = Math.round(1024 * aspectRatio)
      }
    }
    
    // Create canvas with target dimensions
    const canvas = createCanvas(targetWidth, targetHeight)
    const ctx = canvas.getContext('2d')
    
    // Draw the resized image
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight)
    
    // Convert to PNG with high quality but reasonable compression
    const pngBuffer = canvas.toBuffer('image/png', { 
      compressionLevel: 6 // Balance between quality and size
    })
    
    // Check if the PNG is still too large
    if (pngBuffer.length > maxSizeBytes) {
      // Further reduce quality by scaling down more
      const scaleFactor = Math.sqrt(maxSizeBytes / pngBuffer.length) * 0.9 // 10% buffer
      targetWidth = Math.round(targetWidth * scaleFactor)
      targetHeight = Math.round(targetHeight * scaleFactor)
      
      const smallCanvas = createCanvas(targetWidth, targetHeight)
      const smallCtx = smallCanvas.getContext('2d')
      smallCtx.drawImage(img, 0, 0, targetWidth, targetHeight)
      
      const finalBuffer = smallCanvas.toBuffer('image/png', { 
        compressionLevel: 9 // Maximum compression
      })
      
      return `data:image/png;base64,${finalBuffer.toString('base64')}`
    }
    
    console.log(`[edit-agent] Final processed image size: ${pngBuffer.length} bytes`)
    return `data:image/png;base64,${pngBuffer.toString('base64')}`
  } catch (error) {
    console.error('[edit-agent] Image preprocessing failed:', error)
    const errorMessage = error instanceof Error ? error.message : String(error)
    throw new Error(`Failed to preprocess image: ${errorMessage}`)
  }
}

export async function POST(req: NextRequest) {
  console.log("[edit-agent] Received request")
  console.log("[edit-agent] Model server URL:", MODEL_SERVER_URL)
  console.log("[edit-agent] Environment check:")
  console.log(`  - OPENAI_API_KEY: ${OPENAI_API_KEY ? `${OPENAI_API_KEY.substring(0, 7)}...` : 'NOT SET'}`)
  console.log(`  - STABILITY_API_KEY: ${STABILITY_API_KEY ? `${STABILITY_API_KEY.substring(0, 7)}...` : 'NOT SET'}`)
  console.log(`  - NODE_ENV: ${process.env.NODE_ENV}`)
  
  try {
    // Check if model server is reachable
    try {
      const healthCheck = await fetch(`${MODEL_SERVER_URL}/health`, { 
        method: "GET",
        signal: AbortSignal.timeout(10000), // 10 second timeout for health check
      })
      if (!healthCheck.ok) {
        console.error("[edit-agent] Model server health check failed:", healthCheck.status)
        return NextResponse.json({ error: "Model server not responding" }, { status: 503 })
      }
      console.log("[edit-agent] Model server health check passed")
    } catch (error) {
      console.error("[edit-agent] Model server health check error:", error)
      return NextResponse.json({ error: "Cannot connect to model server" }, { status: 503 })
    }
    
    const body = await req.json()
    console.log("[edit-agent] Request body keys:", Object.keys(body))
    console.log("[edit-agent] Request body details:")
    console.log(`  - imageData length: ${body.imageData?.length || 0} chars`)
    console.log(`  - instruction: "${body.instruction}"`)
    console.log(`  - aiModel: "${body.aiModel}"`)
    console.log(`  - previewOnly: ${body.previewOnly}`)
    
    const { imageData, instruction, aiModel, previewOnly } = body as EditAgentInput
    if (!imageData || !instruction || !aiModel) {
      console.error("[edit-agent] Missing required fields:", { 
        hasImageData: !!imageData, 
        hasInstruction: !!instruction, 
        hasAiModel: !!aiModel 
      })
      return NextResponse.json({ error: "Missing imageData, instruction, or aiModel" }, { status: 400 })
    }

    console.log("[edit-agent] Processing:", { instruction, aiModel, previewOnly })
    console.log(`[edit-agent] Image data format: ${imageData.substring(0, 50)}...`)
    const steps: StepLog[] = []
    const { imagePath } = await saveTempImage(imageData)
    steps.push({ step: "upload", info: { imagePath } })

    // Analyze intent and targets
    const intent = parseIntent(instruction)
    console.log("[edit-agent] Parsed intent:", intent)
    steps.push({ step: "analyze", info: intent })
    
    // Validate model compatibility with task
    if (!validateModelForTask(aiModel, intent)) {
      console.warn(`[edit-agent] Model ${aiModel} may not be optimal for this task. Consider using a different model.`)
      steps.push({ step: "model_warning", info: { message: `Model ${aiModel} may not be optimal for this task` } })
    }

    // If lighting/tone -> local enhance
    if (intent.isLighting && !intent.isRemove && !intent.isReplace) {
      console.log("[edit-agent] Processing lighting enhancement")
      // If face-related, get face mask and apply targeted enhancement
      let maskPng: string | undefined = undefined
      if (intent.mentionsFace) {
        try {
          console.log("[edit-agent] Getting face mask")
          const fp = await callFaceParse(imageData)
          maskPng = fp.face_mask_png
          steps.push({ step: "face_parse", image: maskPng })
        } catch (error) {
          console.error("[edit-agent] Face parse failed:", error)
        }
      }
      console.log("[edit-agent] Enhancing image locally")
      const enh = await enhanceLocal(imagePath, imageData, { gamma: 0.9, mask_png: maskPng })
      steps.push({ step: "enhance_local", image: enh.processedImageData })
      return NextResponse.json({ ok: true, result: enh.processedImageData, steps })
    }

    // Else we need a mask
    console.log("[edit-agent] Building mask for target:", intent.target || "object")
    console.log("[edit-agent] Calling model server for mask generation...")
    
    let mask: any
    try {
      mask = await buildMask(imagePath, imageData, intent.target || "object", { dilate: 3, feather: 3 })
      console.log("[edit-agent] Mask generation response received")
      console.log("[edit-agent] Mask response keys:", Object.keys(mask))
      
      if (!mask.mask_png) {
        console.error("[edit-agent] Failed to build mask - no mask_png field")
        console.error("[edit-agent] Mask response:", mask)
        return NextResponse.json({ error: "Failed to build mask - no mask_png field" }, { status: 502 })
      }
      
      // Validate mask data
      if (!mask.mask_png.startsWith('data:image/')) {
        console.error("[edit-agent] Mask is not in expected data URL format")
        console.error("[edit-agent] Mask format:", mask.mask_png.substring(0, 100))
        return NextResponse.json({ error: "Invalid mask format" }, { status: 502 })
      }
      
      console.log("[edit-agent] Mask built successfully, pixels:", mask.mask_binary_sum)
      console.log("[edit-agent] Mask data URL length:", mask.mask_png.length)
      console.log("[edit-agent] Mask format:", mask.mask_png.substring(0, 50))
      steps.push({ step: "mask", info: { pixels: mask.mask_binary_sum }, image: mask.mask_png })
      
    } catch (maskError) {
      console.error("[edit-agent] Mask generation failed:", maskError)
      const errorMessage = maskError instanceof Error ? maskError.message : String(maskError)
      const errorStack = maskError instanceof Error ? maskError.stack : 'No stack trace'
      console.error("[edit-agent] Mask error stack:", errorStack)
      return NextResponse.json({ error: `Mask generation failed: ${errorMessage}` }, { status: 502 })
    }
    
    // Try matting refine to improve edges
    let refinedMask = mask.mask_png
    try {
      console.log("[edit-agent] Attempting matting refine...")
      const r = await callMattingRefine(imageData, mask.mask_png)
      if (r?.refined_mask_png) {
        refinedMask = r.refined_mask_png
        console.log("[edit-agent] Matting refine successful, using refined mask")
        steps.push({ step: "matting_refine", image: refinedMask })
      } else {
        console.log("[edit-agent] Matting refine returned no refined mask, using original")
      }
    } catch (mattingError) {
      console.log("[edit-agent] Matting refine failed, using original mask:", mattingError instanceof Error ? mattingError.message : String(mattingError))
    }

    // Routing and execution
    let edited: { processedImageData: string } | null = null
    if (previewOnly) {
      // Preview: use lightweight fallback inpainting on server
      const editedFb = await callModelServer("inpaint_fallback", { image_data: imageData, mask_png: refinedMask }) as { processedImageData: string }
      steps.push({ step: "inpaint_preview", image: editedFb.processedImageData })
      
      // Try validation but don't fail if it doesn't work
      try {
        const val = await callModelServer("validate_edit", {
          original_image_data: imageData,
          edited_image_data: editedFb.processedImageData,
          mask_png: refinedMask,
          concept: intent.target || "object",
        })
        steps.push({ step: "validate", info: val })
      } catch (validationError) {
        console.error(`[edit-agent] Preview validation failed:`, validationError)
        steps.push({ step: "validate", info: { error: "Validation failed", details: validationError instanceof Error ? validationError.message : String(validationError) } })
      }
      
      return NextResponse.json({ ok: true, result: editedFb.processedImageData, steps })
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
        let val = null
        try {
          val = await callModelServer("validate_edit", {
            original_image_data: imageData,
            edited_image_data: out.processedImageData,
            mask_png: refinedMask,
            concept: intent.target || "object",
          })
        } catch (validationError) {
          console.error(`[edit-agent] SDXL validation failed:`, validationError)
          val = { passed: false, error: "Validation failed" }
        }
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
      // Generate a more specific prompt for OpenAI
      let prompt: string
      if (intent.isReplace && intent.replacement) {
        prompt = `Replace the ${intent.target} with a ${intent.replacement}, maintaining the same style and quality as the original image`
      } else if (intent.isRemove) {
        prompt = `Remove the ${intent.target} completely, fill the area naturally to match the surrounding environment`
      } else {
        prompt = `Edit the ${intent.target} according to the instruction: ${instruction}`
      }
      
      console.log(`[edit-agent] Generated OpenAI prompt: "${prompt}"`)
      console.log(`[edit-agent] Starting OpenAI image edit...`)
      console.log(`[edit-agent] - Original image size: ${imageData.length} chars`)
      console.log(`[edit-agent] - Mask size: ${refinedMask.length} chars`)
      console.log(`[edit-agent] - Target: ${intent.target}`)
      
      try {
        const out = await openAIImagesEdit(imageData, refinedMask, prompt)
        console.log(`[edit-agent] OpenAI edit successful, result size: ${out.processedImageData.length} chars`)
        steps.push({ step: "openai_images_edit", image: out.processedImageData })
        
        console.log(`[edit-agent] Attempting to validate edit result...`)
        let validationResult = null
        try {
          const val = await callModelServer("validate_edit", {
            original_image_data: imageData,
            edited_image_data: out.processedImageData,
            mask_png: refinedMask,
            concept: intent.target || "object",
          })
          validationResult = val
          steps.push({ step: "validate", info: val })
          console.log(`[edit-agent] Validation complete:`, val)
        } catch (validationError) {
          console.error(`[edit-agent] Validation failed, but edit was successful:`, validationError)
          const errorMessage = validationError instanceof Error ? validationError.message : String(validationError)
          steps.push({ step: "validate", info: { error: "Validation failed", details: errorMessage } })
          // Don't fail the entire request if validation fails
        }
        
        return NextResponse.json({ ok: true, result: out.processedImageData, steps })
      } catch (openaiError) {
        console.error(`[edit-agent] OpenAI edit failed:`, openaiError)
        const errorMessage = openaiError instanceof Error ? openaiError.message : String(openaiError)
        throw new Error(`OpenAI image edit failed: ${errorMessage}`)
      }
    }

    if (aiModel === "StabilityAI") {
      const prompt = intent.isReplace && intent.replacement
        ? `Replace ${intent.target} with ${intent.replacement}`
        : `Remove ${intent.target}`
      const out = await stabilityInpaint(imageData, refinedMask, prompt)
      steps.push({ step: "stability_inpaint", image: out.processedImageData })
      let val = null
      try {
        val = await callModelServer("validate_edit", {
          original_image_data: imageData,
          edited_image_data: out.processedImageData,
          mask_png: refinedMask,
          concept: intent.target || "object",
        })
        steps.push({ step: "validate", info: val })
      } catch (validationError) {
        console.error(`[edit-agent] StabilityAI validation failed:`, validationError)
        steps.push({ step: "validate", info: { error: "Validation failed", details: validationError instanceof Error ? validationError.message : String(validationError) } })
      }
      return NextResponse.json({ ok: true, result: out.processedImageData, steps })
    }

    return NextResponse.json({ error: "Unsupported aiModel" }, { status: 400 })
  } catch (e: any) {
    console.error("[edit-agent] Error in POST handler:", e)
    console.error("[edit-agent] Error stack:", e?.stack)
    console.error("[edit-agent] Error type:", typeof e)
    console.error("[edit-agent] Error constructor:", e?.constructor?.name)
    
    // Log additional error details
    if (e instanceof Error) {
      console.error("[edit-agent] Error name:", e.name)
      console.error("[edit-agent] Error message:", e.message)
    }
    
    return NextResponse.json({ 
      error: e?.message || "Unknown error", 
      details: e?.toString(),
      stack: e?.stack,
      type: typeof e,
      constructor: e?.constructor?.name
    }, { status: 500 })
  }
}