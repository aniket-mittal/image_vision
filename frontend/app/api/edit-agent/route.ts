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

async function saveTempDataUrl(dataUrl: string, basename: string) {
  const tempDir = join(tmpdir(), "image_vision_edit")
  await mkdir(tempDir, { recursive: true })
  const base64 = dataUrl.split(",")[1]
  const buf = Buffer.from(base64, "base64")
  const outPath = join(tempDir, `${Date.now()}_${basename}`)
  await writeFile(outPath, buf)
  
  // Also mirror into workspace local temp folder for easy inspection
  try {
    const workspaceTempDir = join(process.cwd(), "temp")
    await mkdir(workspaceTempDir, { recursive: true })
    const wsOutPath = join(workspaceTempDir, `${Date.now()}_${basename}`)
    await writeFile(wsOutPath, buf)
  } catch {}
  return outPath
}

async function saveTempBuffer(buf: Buffer, basename: string) {
  const tempDir = join(tmpdir(), "image_vision_edit")
  await mkdir(tempDir, { recursive: true })
  const outPath = join(tempDir, `${Date.now()}_${basename}`)
  await writeFile(outPath, buf)
  
  // Also mirror into workspace local temp folder for easy inspection
  try {
    const workspaceTempDir = join(process.cwd(), "temp")
    await mkdir(workspaceTempDir, { recursive: true })
    const wsOutPath = join(workspaceTempDir, `${Date.now()}_${basename}`)
    await writeFile(wsOutPath, buf)
  } catch {}
  return outPath
}
async function standardizeMaskAndGetCoverage(maskDataUrl: string, targetW: number, targetH: number) {
  try {
    const srcBuf = Buffer.from(maskDataUrl.split(',')[1] || '', 'base64')
    const standardized = await buildTransparentMaskPng(srcBuf, targetW, targetH)
    // compute coverage on standardized mask
    const alphaForStats = await sharp(standardized).ensureAlpha().extractChannel('alpha').linear(-1, 255).toBuffer()
    const { data } = await sharp(alphaForStats).raw().toBuffer({ resolveWithObject: true }) as any
    let white = 0
    for (let i = 0; i < data.length; i++) if (data[i] === 255) white++
    return { standardized, white }
  } catch (e) {
    console.warn('[edit-agent] standardizeMaskAndGetCoverage failed:', e)
    return { standardized: null as any, white: 0 }
  }
}


async function buildTransparentMaskPng(maskBuffer: Buffer, width?: number, height?: number): Promise<Buffer> {
  // Standardize mask to: alpha = 0 in edit region, 255 elsewhere
  const meta0 = await sharp(maskBuffer).metadata()
  const hasAlpha = (meta0.channels || 0) >= 4 && !!meta0.hasAlpha
  const w = width || meta0.width || 1024
  const h = height || meta0.height || 1024

  let finalAlpha: Buffer
  if (hasAlpha) {
    // Extract alpha once
    const alphaResized = await sharp(maskBuffer)
      .ensureAlpha()
      .extractChannel('alpha')
      .resize(w, h, { fit: 'fill' })
      .toBuffer()

    // Candidate A: invert alpha (hole where object alpha exists)
    const editA = await sharp(alphaResized).linear(-1, 255).blur(0.6).threshold(128).toBuffer()
    const { data: rawA } = await sharp(editA).raw().toBuffer({ resolveWithObject: true }) as any
    let whiteA = 0
    for (let i = 0; i < rawA.length; i++) if (rawA[i] === 255) whiteA++

    // Candidate B: use alpha as-is as white=edit
    const editB = await sharp(alphaResized).blur(0.6).threshold(128).toBuffer()
    const { data: rawB } = await sharp(editB).raw().toBuffer({ resolveWithObject: true }) as any
    let whiteB = 0
    for (let i = 0; i < rawB.length; i++) if (rawB[i] === 255) whiteB++

    // Pick candidate with plausible coverage (not 0, not near total). Prefer the smaller plausible region.
    const total = w * h
    const goodA = whiteA > 0 && whiteA < total * 0.85
    const goodB = whiteB > 0 && whiteB < total * 0.85
    console.log(`[edit-agent] buildTransparentMaskPng(alpha): whiteA=${whiteA} whiteB=${whiteB} total=${total}`)

    // Save candidates for debugging
    try {
      const tmpA = await sharp({ create: { width: w, height: h, channels: 3, background: { r: 255, g: 255, b: 255 } } }).joinChannel(await sharp(editA).linear(-1, 255).toBuffer()).png().toBuffer()
      await saveTempBuffer(tmpA, 'candidate_mask_A_invert_alpha.png')
      const tmpB = await sharp({ create: { width: w, height: h, channels: 3, background: { r: 255, g: 255, b: 255 } } }).joinChannel(await sharp(editB).linear(-1, 255).toBuffer()).png().toBuffer()
      await saveTempBuffer(tmpB, 'candidate_mask_B_as_is_alpha.png')
    } catch {}

    let chosen: Buffer
    if (goodA && goodB) {
      chosen = (whiteA <= whiteB) ? editA : editB
    } else if (goodA) {
      chosen = editA
    } else if (goodB) {
      chosen = editB
    } else {
      // Fallback: pick the one with any coverage; if both invalid, use editA
      chosen = whiteA > 0 ? editA : (whiteB > 0 ? editB : editA)
    }
    // Final alpha is inverse of white=edit
    finalAlpha = await sharp(chosen).linear(-1, 255).toBuffer()
  } else {
    // Grayscale assumed white=edit
    const gray = await sharp(maskBuffer).toColourspace('b-w').resize(w, h, { fit: 'fill' }).blur(0.6).threshold(128).toBuffer()
    finalAlpha = await sharp(gray).linear(-1, 255).toBuffer()
  }

  // Debug: compute coverage on final alpha
  try {
    const { data, info } = await sharp(finalAlpha).raw().toBuffer({ resolveWithObject: true }) as any
    let countWhite = 0
    for (let i = 0; i < data.length; i++) if (data[i] === 0) countWhite++ // finalAlpha is 0 in edit region
    console.log(`[edit-agent] buildTransparentMaskPng: edit pixels (alpha==0)=${countWhite} of ${w*h}`)
  } catch {}

  const rgba = await sharp({ create: { width: w, height: h, channels: 3, background: { r: 0, g: 0, b: 0 } } })
    .joinChannel(finalAlpha)
    .png()
    .toBuffer()
  return rgba
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
    dilate_px: opts?.dilate ?? 2,
    feather_px: opts?.feather ?? 2,
    // Enable advanced refinement
    enable_fba_refine: true,
    enable_controlnet: false, // Disable for edit mode to avoid conflicts
    return_rgba: true,
    keep_largest: true,
    min_area_frac: 0.002,
  }
  // Debug log on the server side for mask coverage
  try {
    const res = await callModelServer("mask_from_text", body) as any
    if (res?.mask_binary_sum !== undefined && Array.isArray(res?.mask_binary_shape)) {
      console.log(`[edit-agent] mask_from_text returned sum=${res.mask_binary_sum} shape=${res.mask_binary_shape}`)
    } else {
      console.log(`[edit-agent] mask_from_text returned keys:`, Object.keys(res || {}))
    }
    return res
  } catch (e) {
    console.error(`[edit-agent] mask_from_text failed:`, e)
    throw e
  }
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
    // forward optional region mask when provided for targeted enhancement
    ...(params?.mask_png ? { mask_png: params.mask_png } : {}),
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
    
    // Preprocess image to meet OpenAI requirements (mask will be handled separately to preserve alpha)
    const processedImage = await preprocessImageForOpenAI(imageData, 4 * 1024 * 1024);
    const imageBuffer = Buffer.from(processedImage.split(',')[1], 'base64');

    // Compute original dimensions once and reuse
    const imageMetadata = await sharp(imageBuffer).metadata();
    const targetW = imageMetadata.width || 1024;
    const targetH = imageMetadata.height || Math.round(targetW / 1.0);

    // Build transparent PNG mask using server-provided maskPng WITHOUT canvas preprocessing to keep alpha
    let sourceMaskBuf: Buffer
    if (maskPng.startsWith('data:image')) {
      sourceMaskBuf = Buffer.from(maskPng.split(',')[1], 'base64')
    } else {
      sourceMaskBuf = Buffer.from(maskPng, 'base64')
    }

    // Always standardize the mask to "transparent where to edit"
    // IMPORTANT: Use server-provided debug binary mask (white=edit), convert to transparent-hole
    let transparentMask: Buffer
    try {
      // Use binary debug if available first
      const serverDebugBinary = maskPng
      // If it's a data URL string, convert to Buffer first
      const maybeBuf = serverDebugBinary.startsWith('data:image')
        ? Buffer.from(serverDebugBinary.split(',')[1], 'base64')
        : Buffer.from(serverDebugBinary, 'base64')
      const dbgMeta = await sharp(maybeBuf).metadata().catch(() => null)
      let dbgBuf: Buffer
      if (dbgMeta) {
        dbgBuf = maybeBuf
      } else {
        dbgBuf = sourceMaskBuf
      }
      // Convert binary white=edit to alpha hole
      const binaryResized = await sharp(dbgBuf).toColourspace('b-w').resize(targetW, targetH, { fit: 'fill' }).toBuffer()
      // Ensure mask is binary: threshold, then invert to hole
      const binaryClean = await sharp(binaryResized).threshold(128).toBuffer()
      const alphaHole = await sharp(binaryClean).linear(-1, 255).toBuffer() // alpha=255 outside, 0 inside
      transparentMask = await sharp({ create: { width: targetW, height: targetH, channels: 3, background: { r: 0, g: 0, b: 0 } } })
        .joinChannel(alphaHole)
        .png()
        .toBuffer()
      console.log('[edit-agent] Built transparent-hole mask from server debug binary')
    } catch (e) {
      console.warn('[edit-agent] Failed to build from debug binary; falling back to standardize path:', e)
      transparentMask = await buildTransparentMaskPng(sourceMaskBuf, targetW, targetH)
    }

    // For statistics, analyze the binary edit mask used to create alpha (white=edit)
    async function computeMaskStats(buf: Buffer) {
      const alphaBuf = await sharp(buf).ensureAlpha().extractChannel('alpha').linear(-1, 255).toBuffer()
      const { data, info } = await sharp(alphaBuf).raw().toBuffer({ resolveWithObject: true }) as any
      let white = 0, black = 0
      // alphaBuf is single-channel
      for (let i = 0; i < data.length; i++) {
        const v = data[i]
        if (v === 255) white++
        else if (v === 0) black++
      }
      return { white, black }
    }
    let { white: whitePixels, black: blackPixels } = await computeMaskStats(transparentMask)
    console.log(`Mask dimensions: ${targetW}x${targetH}`)
    console.log(`Mask pixel analysis: White(edit): ${whitePixels}, Black(keep): ${blackPixels}`)

    // Fallbacks if coverage is zero (no edit region detected)
    if (whitePixels === 0) {
      console.warn('[edit-agent] Zero edit coverage after standardization; trying raw RGBA alpha path')
      try {
        const rawAlphaMask = await sharp(sourceMaskBuf).ensureAlpha().resize(targetW, targetH, { fit: 'fill' }).png().toBuffer()
        let stats = await computeMaskStats(rawAlphaMask)
        console.log(`[edit-agent] Raw alpha path stats: White(edit)=${stats.white}, Black=${stats.black}`)
        if (stats.white > 0) {
          transparentMask = rawAlphaMask
          whitePixels = stats.white
          blackPixels = stats.black
        } else {
          console.warn('[edit-agent] Raw alpha also zero; trying explicit inversion of alpha')
          const invAlpha = await sharp(rawAlphaMask).ensureAlpha().extractChannel('alpha').linear(-1, 255).toBuffer()
          const invRGBA = await sharp({ create: { width: targetW, height: targetH, channels: 3, background: { r: 0, g: 0, b: 0 } } })
            .joinChannel(invAlpha)
            .png()
            .toBuffer()
          stats = await computeMaskStats(invRGBA)
          console.log(`[edit-agent] Inverted alpha stats: White(edit)=${stats.white}, Black=${stats.black}`)
          if (stats.white > 0) {
            transparentMask = invRGBA
            whitePixels = stats.white
            blackPixels = stats.black
          }
        }
      } catch (e) {
        console.warn('[edit-agent] Fallback rebuilds failed:', e)
      }
    }

      // Save debug artifacts
    try {
      await saveTempBuffer(imageBuffer, 'openai_input_image.png')
      await saveTempBuffer(sourceMaskBuf, 'openai_binary_mask_input.png')
      await saveTempBuffer(transparentMask, 'openai_transparent_mask.png')
      // Also save server-provided debug binary, if available
      try {
        const maskedInfo = await callModelServer("mask_from_text", { image_data: imageData, prompt: "__noop__" })
        if (maskedInfo?.debug_binary_mask_png) {
          await saveTempDataUrl(maskedInfo.debug_binary_mask_png, 'openai_transparent_mask_binary_debug.png')
        }
      } catch {}
      // Save alpha channel and binary for visual inspection
      try {
        const alphaCh = await sharp(transparentMask).ensureAlpha().extractChannel('alpha').toBuffer()
        await saveTempBuffer(alphaCh, 'openai_transparent_mask_alpha.png')
      } catch {}
      const workspaceTempDir = join(process.cwd(), 'temp')
      await mkdir(workspaceTempDir, { recursive: true })
      await writeFile(join(workspaceTempDir, `${Date.now()}_openai_input_image.png`), imageBuffer)
      await writeFile(join(workspaceTempDir, `${Date.now()}_openai_mask_transparent.png`), transparentMask)
    } catch (e) {
      console.warn('Failed saving OpenAI debug artifacts:', e)
    }

    // Create FormData for OpenAI API
    const formData = new FormData();
    formData.append('image', new Blob([imageBuffer], { type: 'image/png' }), 'image.png');
    formData.append('mask', new Blob([transparentMask], { type: 'image/png' }), 'mask.png');
    formData.append('model', 'gpt-image-1');
    formData.append('prompt', prompt);
    formData.append('n', '1');
    formData.append('size', '1024x1024');

    console.log("Sending request to OpenAI with:");
    console.log(`- Image size: ${imageBuffer.length} bytes`);
    console.log(`- Mask size: ${transparentMask.length} bytes`);
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
        'Accept': 'application/json',
      },
      body: formData,
      duplex: 'half',
      signal: AbortSignal.timeout(120000),
    } as any);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`OpenAI API error: ${response.status} - ${errorText}`);
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    console.log("OpenAI API response:", data);

    let processedImageData: string;

    if (data.data?.[0]?.b64_json) {
      processedImageData = `data:image/png;base64,${data.data[0].b64_json}`;
      console.log("OpenAI edit successful using b64_json");
    } else if (data.data?.[0]?.url) {
      console.log("Found URL in response, converting to base64");
      const imageResponse = await fetch(data.data[0].url);
      if (!imageResponse.ok) {
        throw new Error(`Failed to download image from OpenAI URL: ${imageResponse.status}`);
      }
      const arrBuf = await imageResponse.arrayBuffer();
      const base64 = Buffer.from(arrBuf).toString('base64');
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

      // Build overlay alpha that is 255 INSIDE edit region (so generated content only paints inside hole)
      const baseAlpha = await sharp(transparentMask)
        .ensureAlpha()
        .extractChannel('alpha')
        .resize(targetWidth, targetHeight, { fit: 'fill' })
        .toBuffer();
      // Build a multi-scale feathered alpha to reduce seams
      let editAlpha = await sharp(baseAlpha).linear(-1, 255).toBuffer()
      // Heavier multi-scale blur + lower threshold to expand the edit region (~another 10%)
      editAlpha = await sharp(editAlpha).blur(3.0).toBuffer()
      editAlpha = await sharp(editAlpha).blur(1.4).toBuffer()
      editAlpha = await sharp(editAlpha).threshold(48).toBuffer()

      const genRGBA = await sharp(genResized)
        .joinChannel(editAlpha)
        .png()
        .toBuffer();

      const originalResized = await sharp(imageBuffer)
        .resize(targetWidth, targetHeight, { fit: 'fill' })
        .toBuffer();

      const finalComposite = await sharp(originalResized)
        .composite([{ input: genRGBA, blend: 'over' }])
        // Strip any stray metadata and trim tiny 1px transparent borders if present
        .png({ progressive: false })
        .toBuffer();

      processedImageData = `data:image/png;base64,${finalComposite.toString('base64')}`;
      console.log(`Composited generated content into EDIT region at ${targetWidth}x${targetHeight}`);

      // Seamless blend on server for additional smoothing and realism
      try {
        const sb = await callModelServer("seamless_blend", {
          original_image_data: imageData,
          edited_image_data: processedImageData,
          mask_png: maskPng,
          feather_px: 30,
          pyramid_levels: 5,
          add_grain: false,
          expand_px: 24,
        }) as { processedImageData: string }
        if (sb?.processedImageData) {
          processedImageData = sb.processedImageData
          console.log("Applied server seamless_blend post-process")
        }
      } catch (e) {
        console.warn("seamless_blend failed or skipped:", e)
      }
    }

    // Save result for debugging
    try {
      await saveTempDataUrl(processedImageData, 'openai_result.png')
    } catch (e) {
      console.warn('Failed saving OpenAI result artifact:', e)
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
    // Optional tuning for better photorealism
    form.append("output_format", "png")
    
    // Save debug artifacts
    try {
      await saveTempBuffer(imageBuf, 'stability_input_image.png')
      await saveTempBuffer(maskBuf, 'stability_input_mask.png')
    } catch (e) {
      console.warn('Failed saving Stability debug artifacts:', e)
    }

    const r = await fetch("https://api.stability.ai/v2beta/stable-image/edit/inpaint", {
      method: "POST",
      headers: { 
        Authorization: `Bearer ${STABILITY_API_KEY}`,
        Accept: "application/json, image/*"
      },
      body: form as any,
      duplex: 'half',
      signal: AbortSignal.timeout(120000),
    } as any)
    
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
    const outUrl = `data:image/png;base64,${b64}`
    try { await saveTempDataUrl(outUrl, 'stability_result.png') } catch {}
    return { processedImageData: outUrl }
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
      // If face-related, get face mask; else if a target is mentioned, build a target mask
      let maskPng: string | undefined = undefined
      try {
        if (intent.mentionsFace) {
          console.log("[edit-agent] Getting face mask")
          const fp = await callFaceParse(imageData)
          maskPng = fp.face_mask_png
          steps.push({ step: "face_parse", image: maskPng })
        } else if (intent.target && intent.target !== 'object') {
          console.log("[edit-agent] Getting target mask for lighting:", intent.target)
          const targetMask = await buildMask(imagePath, imageData, intent.target, { dilate: 1, feather: 3 })
          maskPng = targetMask.mask_png
          steps.push({ step: "lighting_target_mask", image: maskPng })
        }
      } catch (e) {
        console.warn("[edit-agent] Target mask for lighting failed, proceeding without:", e)
      }

      // Map instruction to enhancement parameters
      const t = instruction.toLowerCase()
      const wantsBrighter = /(brighten|brighter|increase exposure|raise exposure|lighter|lighten|illuminate)/.test(t)
      const wantsDarker = /(darken|darker|decrease exposure|lower exposure|dim|shade)/.test(t)
      const wantsContrast = /(contrast|clarity|punch|vibrance|saturation)/.test(t)
      const wantsDenoise = /(denoise|noise|grain|smooth|clean)/.test(t)
      const wantsWhiteBalance = /(white balance|color balance|temperature|tint|warm|cool|neutral)/.test(t)
      const wantsHighlights = /(highlights|specular|glow|shine|recover highlights|reduce highlights|enhance highlights)/.test(t)
      const wantsShadows = /(shadows|shadow detail|dark areas|lift shadows|recover shadows|deepen shadows)/.test(t)

      const params = {
        gamma: wantsBrighter ? 0.8 : wantsDarker ? 1.15 : 0.95,
        clahe: wantsContrast || wantsHighlights || wantsShadows ? true : false,
        unsharp_amount: wantsContrast ? 0.45 : wantsHighlights ? 0.25 : 0.15,
        unsharp_radius: wantsContrast ? 4 : wantsHighlights ? 3 : 2,
        denoise: wantsDenoise,
        white_balance: wantsWhiteBalance,
        // Add advanced lighting controls
        highlights: wantsHighlights ? (wantsHighlights && wantsContrast ? 0.35 : 0.25) : 0.0,
        shadows: wantsShadows ? (wantsShadows && wantsContrast ? 0.45 : 0.35) : 0.0,
        vibrance: wantsContrast ? 0.25 : 0.0,
        mask_png: maskPng,
      }
      console.log("[edit-agent] Enhancing image locally with params:", params)
      const enh = await enhanceLocal(imagePath, imageData, params)
      steps.push({ step: "enhance_local", image: enh.processedImageData })
      return NextResponse.json({ ok: true, result: enh.processedImageData, steps })
    }

    // Else we need a mask
    console.log("[edit-agent] Building mask for target:", intent.target || "object")
    console.log("[edit-agent] Calling model server for mask generation...")
    
    let mask: any
    try {
      // Use more precise masking for object removal
      mask = await buildMask(imagePath, imageData, intent.target || "object", { dilate: 1, feather: 2 })
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
    
    // Build canonical OpenAI mask from server debug binary if available
    const openAIMask = mask?.debug_binary_mask_png || mask?.mask_png
    steps.push({ step: "mask_openai_source", image: openAIMask })
    try { await saveTempDataUrl(openAIMask, 'final_edit_mask.png') } catch {}

    // Routing and execution
    let edited: { processedImageData: string } | null = null
    if (previewOnly) {
      // Preview: use lightweight fallback inpainting on server
      const editedFb = await callModelServer("inpaint_fallback", { image_data: imageData, mask_png: openAIMask }) as { processedImageData: string }
      steps.push({ step: "inpaint_preview", image: editedFb.processedImageData })
      
      // Try validation but don't fail if it doesn't work
      try {
      const val = await callModelServer("validate_edit", {
        original_image_data: imageData,
        edited_image_data: editedFb.processedImageData,
        mask_png: openAIMask,
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
      // Optimize SDXL parameters for object removal and natural fill
      let params: any = { 
        guidance_scale: 7.5, 
        num_inference_steps: 40, 
        use_canny: false, // Disable canny for more natural results
        use_depth: false,
        // Add negative prompts to avoid artifacts
        negative_prompt: "low quality, blurry, artifacts, cartoon, illustration, anime, painting, cgi, 3d render, repetitive texture, pattern, watermark, text"
      }
      let attempt = 0
      let best = null as any
      while (attempt < 2) {
        const out = await inpaintSDXL(imageData, openAIMask, prompt, params)
        let val = null
        try {
          val = await callModelServer("validate_edit", {
            original_image_data: imageData,
            edited_image_data: out.processedImageData,
            mask_png: openAIMask,
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
        params = { 
          ...params, 
          guidance_scale: params.guidance_scale + 1.0, 
          num_inference_steps: params.num_inference_steps + 15,
          use_canny: true // Enable canny on retry for better structure
        }
      }
      if (!edited && best) edited = best.out
      return NextResponse.json({ ok: true, result: edited?.processedImageData || null, steps })
    }

    if (aiModel === "OpenAIImages") {
      // Use the user's instruction verbatim for adaptability
      // Enhance prompt for better object removal and natural fill
      const prompt: string = intent.isRemove 
        ? `Remove ${intent.target} completely and fill the area naturally with the surrounding background, making it look like the ${intent.target} was never there`
        : instruction
      
      console.log(`[edit-agent] Generated OpenAI prompt: "${prompt}"`)
      console.log(`[edit-agent] Starting OpenAI image edit...`)
      
      try {
        const out = await openAIImagesEdit(imageData, openAIMask, prompt)
        steps.push({ step: "openai_images_edit", image: out.processedImageData })
        // Optionally run validation without failing the request
        try {
          const val = await callModelServer("validate_edit", {
            original_image_data: imageData,
            edited_image_data: out.processedImageData,
            mask_png: openAIMask,
            concept: intent.target || "object",
          })
          steps.push({ step: "validate", info: val })
        } catch (validationError) {
          steps.push({ step: "validate", info: { error: validationError instanceof Error ? validationError.message : String(validationError) } })
        }
        return NextResponse.json({ ok: true, result: out.processedImageData, steps })
      } catch (openAIError) {
        console.error("OpenAI Images API error:", openAIError);
        return NextResponse.json({ error: `OpenAI API error: ${openAIError instanceof Error ? openAIError.message : String(openAIError)}` }, { status: 502 });
      }
    }

    if (aiModel === "StabilityAI") {
      try {
        const stabilityPrompt: string = intent.isRemove
          ? `Remove ${intent.target} completely and fill the area naturally with the surrounding background`
          : instruction
        const out = await stabilityInpaint(imageData, openAIMask, stabilityPrompt)
        steps.push({ step: "stability_inpaint", image: out.processedImageData })
        return NextResponse.json({ ok: true, result: out.processedImageData, steps })
      } catch (stabilityError) {
        console.error("Stability AI inpainting failed:", stabilityError);
        return NextResponse.json({ error: `Stability AI API error: ${stabilityError instanceof Error ? stabilityError.message : String(stabilityError)}` }, { status: 502 });
      }
    }

    return NextResponse.json({ error: "Unsupported AI model" }, { status: 501 });

  } catch (error) {
    console.error("[edit-agent] Unexpected error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}