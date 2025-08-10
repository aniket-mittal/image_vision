import { NextRequest, NextResponse } from "next/server"
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"
import { createCanvas, loadImage } from "canvas"
import sharp from "sharp";

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
  try {
    // Validate OpenAI API key
    const openaiApiKey = process.env.OPENAI_API_KEY;
    if (!openaiApiKey) {
      throw new Error("OpenAI API key not configured");
    }
    
    // Log warning if key format looks suspicious (but don't fail)
    if (!openaiApiKey.startsWith('sk-')) {
      console.warn("OpenAI API key format looks suspicious - may cause API errors");
    }

    console.log("Preprocessing image and mask for OpenAI");
    
    // Preprocess image to meet OpenAI requirements
    const processedImage = await preprocessImageForOpenAI(imageData, 4 * 1024 * 1024);
    let processedMask = await preprocessImageForOpenAI(maskPng, 4 * 1024 * 1024);
    
    console.log(`Processed image size: ${processedImage.length} chars, mask size: ${processedMask.length} chars`);
    
    // Validate mask format - ensure it's a proper binary mask
    let maskBuffer = Buffer.from(processedMask.split(',')[1], 'base64');
    const maskImage = await sharp(maskBuffer);
    const maskMetadata = await maskImage.metadata();
    
    console.log(`Mask dimensions: ${maskMetadata.width}x${maskMetadata.height}`);
    
    // Analyze mask pixels to ensure it's properly formatted
    const maskPixels = await maskImage.raw().toBuffer();
    let whitePixels = 0, blackPixels = 0, otherPixels = 0;
    
    for (let i = 0; i < maskPixels.length; i++) {
      const pixel = maskPixels[i];
      if (pixel === 255) whitePixels++;
      else if (pixel === 0) blackPixels++;
      else otherPixels++;
    }
    
    console.log(`Mask pixel analysis: White (edit areas): ${whitePixels}, Black (preserve): ${blackPixels}, Other: ${otherPixels}`);
    
    // Ensure mask is binary (only black and white)
    if (otherPixels > 0) {
      console.warn("Mask contains non-binary pixels, converting to binary");
      const binaryMask = await maskImage
        .threshold(128) // Convert to binary
        .png()
        .toBuffer();
      const binaryMaskBase64 = `data:image/png;base64,${binaryMask.toString('base64')}`;
      processedMask = binaryMaskBase64;
      maskBuffer = Buffer.from(processedMask.split(',')[1], 'base64');
    }
    
    // Get original image dimensions for aspect ratio preservation
    const imageBuffer = Buffer.from(processedImage.split(',')[1], 'base64');
    const image = await sharp(imageBuffer);
    const imageMetadata = await image.metadata();
    
    console.log(`Image dimensions: ${imageMetadata.width}x${imageMetadata.height}`);
    
    // Create FormData for OpenAI API
    const formData = new FormData();
    formData.append('image', new Blob([imageBuffer], { type: 'image/png' }), 'image.png');
    formData.append('mask', new Blob([maskBuffer], { type: 'image/png' }), 'mask.png');
    formData.append('prompt', prompt);
    formData.append('n', '1');
    formData.append('size', '1024x1024'); // OpenAI edits always return 1024x1024
    
    console.log("Sending request to OpenAI with:");
    console.log(`- Image size: ${imageBuffer.length} bytes`);
    console.log(`- Mask size: ${maskBuffer.length} bytes`);
    console.log(`- Prompt: "${prompt}"`);
    console.log(`- Size: 1024x1024`);
    
    // Log form data entries for debugging
    for (const [key, value] of formData.entries()) {
      if (value instanceof Blob) {
        console.log(`- Form data entries: ${key}: ${value.type} (${value.size} bytes)`);
      } else {
        console.log(`- Form data entries: ${key}: ${value}`);
      }
    }
    
    const response = await fetch('https://api.openai.com/v1/images/edits', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`OpenAI API error: ${response.status} - ${errorText}`);
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    console.log("OpenAI API response:", data);

    let processedImageData: string;

    if (data.data?.[0]?.b64_json) {
      // Use base64 data if available
      processedImageData = `data:image/png;base64,${data.data[0].b64_json}`;
      console.log("OpenAI edit successful using b64_json");
    } else if (data.data?.[0]?.url) {
      // Download image from URL and convert to base64
      console.log("Found URL in response, converting to base64");
      const imageResponse = await fetch(data.data[0].url);
      if (!imageResponse.ok) {
        throw new Error(`Failed to download image from OpenAI URL: ${imageResponse.status}`);
      }
      
      const imageBuffer = await imageResponse.arrayBuffer();
      const base64 = Buffer.from(imageBuffer).toString('base64');
      processedImageData = `data:image/png;base64,${base64}`;
      console.log("Successfully converted URL to base64");
    } else {
      throw new Error("No image data in response");
    }

    // Post-process: restore original aspect ratio and composite with original using mask
    if (imageMetadata.width && imageMetadata.height) {
      const originalAspectRatio = imageMetadata.width / imageMetadata.height;
      const targetWidth = 1024;
      const targetHeight = Math.round(targetWidth / originalAspectRatio);
      
      const genBuf = Buffer.from(processedImageData.split(',')[1], 'base64');
      const genResized = await sharp(genBuf)
        .resize(targetWidth, targetHeight, { fit: 'fill' })
        .toBuffer();

      // Resize mask to original aspect ratio dimensions and prepare as 8-bit alpha
      const maskAlpha = await sharp(maskBuffer)
        .resize(targetWidth, targetHeight, { fit: 'fill' })
        .threshold(128)
        .toColourspace('b-w')
        .ensureAlpha() // guarantees an alpha channel exists
        .removeAlpha() // keep single channel grayscale
        .toBuffer();

      // Build RGBA for generated image using mask as alpha
      const genRGBA = await sharp(genResized)
        .joinChannel(maskAlpha) // maskAlpha used as alpha channel
        .png()
        .toBuffer();

      // Composite: gen over original using alpha so only masked region is replaced
      const originalResized = await sharp(imageBuffer)
        .resize(targetWidth, targetHeight, { fit: 'fill' })
        .toBuffer();

      const finalComposite = await sharp(originalResized)
        .composite([{ input: genRGBA, blend: 'over' }])
        .png()
        .toBuffer();

      processedImageData = `data:image/png;base64,${finalComposite.toString('base64')}`;
      console.log(`Composited generated content into masked region at ${targetWidth}x${targetHeight}`);
    }

    console.log(`OpenAI edit successful, result size: ${processedImageData.length} chars`);
    return { processedImageData };

  } catch (error) {
    console.error("OpenAI Images API error:", error);
    throw error;
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
        Accept: "application/json, image/*"
      },
      body: form as any,
    })
    
    if (!r.ok) throw new Error(`stability ${r.status}: ${await r.text()}`)
    
    const contentType = r.headers.get("content-type") || ""
    if (contentType.includes("application/json")) {
      const j = await r.json()
      const b64 = j?.artifacts?.[0]?.base64
      if (b64) {
        return { processedImageData: `data:image/png;base64,${b64}` }
      }
      // Fallback: sometimes the API returns an image despite JSON header confusion
      const alt = await fetch("https://api.stability.ai/v2beta/stable-image/edit/inpaint", {
        method: "POST",
        headers: { Authorization: `Bearer ${STABILITY_API_KEY}` },
        body: form as any,
      })
      const arr = new Uint8Array(await alt.arrayBuffer())
      const b64img = Buffer.from(arr).toString("base64")
      if (!b64img) throw new Error("stability: empty result")
      return { processedImageData: `data:image/png;base64,${b64img}` }
    }
    // Image bytes
    const arr = new Uint8Array(await r.arrayBuffer())
    const b64 = Buffer.from(arr).toString("base64")
    return { processedImageData: `data:image/png;base64,${b64}` }
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
      // Use the user's instruction verbatim for adaptability
      const prompt = instruction
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
      // Use the user's instruction verbatim for adaptability
      const prompt: string = instruction
      
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
      const out = await stabilityInpaint(imageData, refinedMask, instruction)
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