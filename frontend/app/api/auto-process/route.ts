import { openai } from "@ai-sdk/openai"
import { streamText } from "ai"
import { NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"

// OpenAI API key
const OPENAI_API_KEY = process.env.OPENAI_API_KEY as string

interface ProcessingResult {
  processedImageData: string | null
  refinedQuery: string
  processingType: "attention" | "segmentation" | "original"
  parameters: any
  answer: string
}

async function generateImageCaption(imageData: string): Promise<string> {
  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Please provide a detailed description of this image. Focus on the main objects, their positions, and any notable details that would be relevant for visual analysis." },
              { type: "image_url", image_url: { url: imageData } }
            ]
          }
        ],
        max_tokens: 200
      })
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("OpenAI API error in generateImageCaption:", response.status, errorText)
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`)
    }

    const result = await response.json()
    
    if (!result.choices || !result.choices[0] || !result.choices[0].message) {
      console.error("Unexpected OpenAI response format in generateImageCaption:", result)
      throw new Error("Unexpected OpenAI response format")
    }

    return result.choices[0].message.content
  } catch (error) {
    console.error("Error in generateImageCaption:", error)
    // Return a fallback description
    return "An image with various objects and elements that may be relevant to the user's question."
  }
}

async function determineProcessingStrategy(userQuery: string, imageCaption: string): Promise<{
  processingType: "attention" | "segmentation" | "original"
  refinedQuery: string
  parameters: any
}> {
  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are an expert at determining the best image processing strategy for visual analysis questions. Based on the user's question and image description, determine:\n\n1. Processing Type:\n   - \"attention\": Use attention masking when the question focuses on specific objects or regions that need to be highlighted\n   - \"segmentation\": Use segmentation when the question involves object boundaries, spatial relationships, or precise object isolation\n   - \"original\": Use the original image when the question is general or doesn't require specific object focus\n\n2. Refined Query: Extract ALL relevant objects/concepts from the user's question. If multiple objects are mentioned or relationships are involved, include all relevant objects separated by "and" (e.g., "birds and elephant" from "what direction are the birds facing in relation to the elephant")\n\n3. Parameters:\n   - For attention: layer_index (23), enhancement_control (5.0), smoothing_kernel (3), overlay_strength (1.0), grayscale_level (200)\n   - For segmentation: blur_strength (15), padding (20), mask_type (\"precise\" or \"oval\")\n\nRespond in JSON format:\n{\n  \"processingType\": \"attention|segmentation|original\",\n  \"refinedQuery\": \"extracted object/concept\",\n  \"parameters\": {\n    // specific parameters based on type\n  }\n}`
          },
          {
            role: "user",
            content: `User Question: \"${userQuery}\"\nImage Description: \"${imageCaption}\"\n\nWhat processing strategy should be used?`
          }
        ],
        max_tokens: 300
      })
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("OpenAI API error:", response.status, errorText)
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`)
    }

    const result = await response.json()
    
    if (!result.choices || !result.choices[0] || !result.choices[0].message) {
      console.error("Unexpected OpenAI response format:", result)
      throw new Error("Unexpected OpenAI response format")
    }

    const content = result.choices[0].message.content
    console.log("OpenAI response content:", content)
    
    try {
      // Remove markdown code blocks if present
      const cleanContent = content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim()
      return JSON.parse(cleanContent)
    } catch (parseError) {
      console.error("Error parsing JSON from OpenAI response:", parseError)
      console.error("Raw content:", content)
      // Fallback to original image processing
      return {
        processingType: "original",
        refinedQuery: userQuery,
        parameters: {}
      }
    }
  } catch (error) {
    console.error("Error in determineProcessingStrategy:", error)
    // Fallback to original image processing
    return {
      processingType: "original",
      refinedQuery: userQuery,
      parameters: {}
    }
  }
}

async function runAttentionMasking(imagePath: string, query: string, parameters: any): Promise<string> {
  // Use persistent model server to avoid reloading CLIP per request
  const requestBody = {
    image_path: imagePath,
    query,
    layer_index: parameters.layer_index ?? 23,
    enhancement_control: parameters.enhancement_control ?? 5.0,
    smoothing_kernel: parameters.smoothing_kernel ?? 3,
    grayscale_level: parameters.grayscale_level ?? 200,
    overlay_strength: parameters.overlay_strength ?? 1.0,
    output_dir: "temp_output",
  }
  const MODEL_SERVER_URL = process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765"
  const res = await fetch(`${MODEL_SERVER_URL}/attention`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  })
  if (!res.ok) {
    throw new Error(`Model server error ${res.status}: ${await res.text()}`)
  }
  const data = await res.json()
  if (!data.saved) throw new Error("Model server did not return saved path")
  return data.saved
}

async function runSegmentationMasking(imagePath: string, query: string, parameters: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const maskType = parameters.mask_type || "precise"
    const cwd = "/Users/aniketmittal/Desktop/code/image_vision"
    const pythonProcess = spawn("conda", [
      "run", "-n", "clip_api", "python",
      join(cwd, "run_segmentation.py"),
      imagePath,
      query,
      "--blur_strength", parameters.blur_strength?.toString() || "15",
      "--padding", parameters.padding?.toString() || "20",
      "--output_dir", "temp_output"
    ], {
      cwd: cwd
    })

    let output = ""
    let errorOutput = ""

    pythonProcess.stdout.on("data", (data) => {
      output += data.toString()
    })

    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString()
    })

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        // Find the output file path based on mask type
        const filePattern = maskType === "oval" ? /Saved oval mask to: (.+\.jpg)/ : /Saved precise mask to: (.+\.jpg)/
        const match = output.match(filePattern)
        if (match) {
          resolve(match[1])
        } else {
          reject(new Error("Could not find output file path"))
        }
      } else {
        reject(new Error(`Process failed with code ${code}: ${errorOutput}`))
      }
    })
  })
}

async function getFinalAnswer(processedImageData: string, userQuery: string, refinedQuery: string, processingType: string): Promise<string> {
  try {
    console.log("Getting final answer from GPT-4V...")
    
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are an expert at analyzing images and providing detailed, accurate answers to questions. 

The user has asked: "${userQuery}"

The image has been processed using ${processingType} technique with the refined query: "${refinedQuery}"

Please provide a comprehensive, detailed answer to the user's question based on the processed image. Be thorough and include all relevant details you can observe. If the question is about specific objects, locations, or relationships in the image, make sure to address those specifically.

If you cannot see the image clearly or if the image appears to be corrupted, please let the user know.`
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: `Please analyze this processed image and answer the following question in detail: "${userQuery}"`
              },
              {
                type: "image_url",
                image_url: {
                  url: processedImageData
                }
              }
            ]
          }
        ],
        max_tokens: 1000
      })
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("OpenAI API error in getFinalAnswer:", response.status, errorText)
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`)
    }

    const result = await response.json()
    const answer = result.choices[0]?.message?.content || "Unable to generate answer"
    
    console.log("Final answer generated successfully")
    return answer

  } catch (error) {
    console.error("Error getting final answer:", error)
    return "Sorry, I encountered an error while analyzing the image. Please try again."
  }
}

async function processImage(imageData: string, userQuery: string): Promise<ProcessingResult> {
  try {
    // Step 1: Generate image caption
    console.log("Generating image caption...")
    const imageCaption = await generateImageCaption(imageData)
    console.log("Image caption:", imageCaption)

    // Step 2: Determine processing strategy
    console.log("Determining processing strategy...")
    console.log("[Auto] Strategy selection start")
    const strategy = await determineProcessingStrategy(userQuery, imageCaption)
    console.log("[Auto] Strategy selection done", strategy)
    console.log("Processing strategy:", strategy)

    // Step 3: Process image based on strategy
    let processedImageData: string | null = null

    if (strategy.processingType === "original") {
      // Use original image
      processedImageData = imageData
    } else {
      // Save image to temporary file
      const tempDir = join(tmpdir(), "image_vision_temp")
      await mkdir(tempDir, { recursive: true })
      
      const imageBuffer = Buffer.from(imageData.split(",")[1], "base64")
      const imagePath = join(tempDir, "input_image.jpg")
      await writeFile(imagePath, imageBuffer)

      // Run appropriate processing
      if (strategy.processingType === "attention") {
        try {
          console.log("[Auto] Attention inference start")
          const outputPath = await runAttentionMasking(imagePath, strategy.refinedQuery, strategy.parameters)
          const absoluteOutputPath = join("/Users/aniketmittal/Desktop/code/image_vision", outputPath)
          const processedImageBuffer = await import("fs/promises").then(fs => fs.readFile(absoluteOutputPath))
          processedImageData = `data:image/jpeg;base64,${processedImageBuffer.toString("base64")}`
          console.log("[Auto] Attention inference done")
        } catch (pythonError) {
          console.error("Python script error in auto-process attention:", pythonError)
          // Fallback to original image if Python processing fails
          processedImageData = imageData
        }
      } else if (strategy.processingType === "segmentation") {
        try {
          console.log("[Auto] Segmentation inference start")
          const outputPath = await runSegmentationMasking(imagePath, strategy.refinedQuery, strategy.parameters)
          const absoluteOutputPath = join("/Users/aniketmittal/Desktop/code/image_vision", outputPath)
          const processedImageBuffer = await import("fs/promises").then(fs => fs.readFile(absoluteOutputPath))
          processedImageData = `data:image/jpeg;base64,${processedImageBuffer.toString("base64")}`
          console.log("[Auto] Segmentation inference done")
        } catch (pythonError) {
          console.error("Python script error in auto-process segmentation:", pythonError)
          // Fallback to original image if Python processing fails
          processedImageData = imageData
        }
      }
    }

    // Step 4: Get final answer from GPT-4V
    console.log("Getting final answer...")
    const answer = await getFinalAnswer(processedImageData!, userQuery, strategy.refinedQuery, strategy.processingType)
    console.log("Answer generated:", answer.substring(0, 100) + "...")

    return {
      processedImageData,
      refinedQuery: strategy.refinedQuery,
      processingType: strategy.processingType,
      parameters: strategy.parameters,
      answer
    }

  } catch (error) {
    console.error("Error in auto processing:", error)
    throw error
  }
}

export async function POST(req: NextRequest) {
  try {
    const { imageData, userQuery } = await req.json()

    if (!imageData || !userQuery) {
      return NextResponse.json({ error: "Missing imageData or userQuery" }, { status: 400 })
    }

    console.log("Starting auto processing...")
    const result = await processImage(imageData, userQuery)
    console.log("Auto processing completed")

    return NextResponse.json(result)

  } catch (error) {
    console.error("API Error:", error)
    return NextResponse.json({ error: "Error processing request" }, { status: 500 })
  }
} 