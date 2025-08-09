import { NextRequest, NextResponse } from "next/server"
// Removed local python fallback – always use model server
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"

// OpenAI API key
const OPENAI_API_KEY = process.env.OPENAI_API_KEY as string

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

async function refineQueryForSegmentation(userQuery: string, imageCaption: string): Promise<{
  refinedQuery: string
  parameters: any
}> {
  try {
    console.log("Calling OpenAI for segmentation refinement...")
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
            content: `You are an expert at refining queries for segmentation mask generation. Your primary goal is to extract the key objects/concepts from the user's question that need to be segmented.

IMPORTANT: Prioritize the user's question over the image description. The user's question contains the specific objects they want to focus on.

Guidelines:
1. Extract ALL relevant objects mentioned in the user's question
2. If the question involves relationships between objects, include both objects
3. Use simple, clear object names separated by "and" if multiple objects
4. Focus on what the user is actually asking about, not what's in the image description
5. Choose mask_type based on the question:
   - "precise": For questions about specific objects, boundaries, or relationships
   - "oval": For questions about general areas or circular objects

Examples:
- "What direction are the birds facing in relation to the elephant?" → "birds and elephant" (precise)
- "How many objects are there?" → "all objects" (precise)
- "Is there a circular area?" → "circular area" (oval)
- "What's the relationship between the cat and dog?" → "cat and dog" (precise)

Respond in JSON format:
{
  "refinedQuery": "extracted objects for segmentation",
  "parameters": {
    "blur_strength": 15,
    "padding": 20,
    "mask_type": "precise"
  }
}`
          },
          {
            role: "user",
            content: `User Question: \"${userQuery}\"\nImage Description: \"${imageCaption}\"\n\nWhat should be the refined query and parameters for segmentation?`
          }
        ],
        max_tokens: 200
      })
    })

    console.log("OpenAI response status:", response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error("OpenAI API error in refineQueryForSegmentation:", response.status, errorText)
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`)
    }

    const result = await response.json()
    console.log("OpenAI response result:", JSON.stringify(result, null, 2))
    
    if (!result.choices || !result.choices[0] || !result.choices[0].message) {
      console.error("Unexpected OpenAI response format in refineQueryForSegmentation:", result)
      throw new Error("Unexpected OpenAI response format")
    }

    const content = result.choices[0].message.content
    console.log("OpenAI response content for segmentation refinement:", content)
    
    try {
      // Remove markdown code blocks if present
      const cleanContent = content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim()
      const parsed = JSON.parse(cleanContent)
      console.log("Successfully parsed JSON:", parsed)
      return parsed
    } catch (parseError) {
      console.error("Error parsing JSON from OpenAI response in refineQueryForSegmentation:", parseError)
      console.error("Raw content:", content)
      // Fallback
      return {
        refinedQuery: userQuery,
        parameters: {
          blur_strength: 15,
          padding: 20,
          mask_type: "precise"
        }
      }
    }
  } catch (error) {
    console.error("Error in refineQueryForSegmentation:", error)
    // Fallback
    return {
      refinedQuery: userQuery,
      parameters: {
        blur_strength: 15,
        padding: 20,
        mask_type: "precise"
      }
    }
  }
}

async function runSegmentationMasking(imagePath: string, query: string, parameters: any): Promise<{ processedImageData?: string; savedPath?: string; backend: "model-server" }> {
  // Call model server only
  const MODEL_SERVER_URL = (process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "")
  const requestBody = {
    image_path: imagePath,
    image_data: await import("fs/promises").then(fs => fs.readFile(imagePath).then(b=>`data:image/jpeg;base64,${b.toString("base64")}`)),
    query,
    blur_strength: parameters.blur_strength ?? 15,
    padding: parameters.padding ?? 20,
    mask_type: parameters.mask_type || "precise",
    output_dir: "temp_output",
  }
  const res = await fetch(`${MODEL_SERVER_URL}/segmentation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  })
  if (!res.ok) {
    throw new Error(`Model server segmentation failed ${res.status}: ${await res.text()}`)
  }
  const data = await res.json()
  if (!data.processedImageData) {
    throw new Error("Model server segmentation did not return processedImageData")
  }
  console.log("[Segmentation] Using model-server backend")
  return { processedImageData: data.processedImageData, backend: "model-server" }
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

export async function POST(req: NextRequest) {
  try {
    const { imageData, userQuery } = await req.json()

    if (!imageData || !userQuery) {
      return NextResponse.json({ error: "Missing imageData or userQuery" }, { status: 400 })
    }

    console.log("[Segmentation] Starting")

    // Step 1: Generate image caption
    const imageCaption = await generateImageCaption(imageData)
    console.log("[Segmentation] Caption done")

    // Step 2: Refine query for segmentation
    const { refinedQuery, parameters } = await refineQueryForSegmentation(userQuery, imageCaption)
    console.log("[Segmentation] Refinement done", { refinedQuery, parameters })

    // Step 3: Run segmentation masking
    const tempDir = join(tmpdir(), "image_vision_temp")
    await mkdir(tempDir, { recursive: true })
    
    const imageBuffer = Buffer.from(imageData.split(",")[1], "base64")
    const imagePath = join(tempDir, "input_image.jpg")
    await writeFile(imagePath, imageBuffer)

    try {
      console.log("[Segmentation] Inference start")
      const result = await runSegmentationMasking(imagePath, refinedQuery, parameters)
      let processedImageData: string
      if (result.processedImageData) {
        processedImageData = result.processedImageData
      } else if (result.savedPath) {
        const absoluteOutputPath = join("/Users/aniketmittal/Desktop/code/image_vision", result.savedPath)
        const processedImageBuffer = await import("fs/promises").then(fs => fs.readFile(absoluteOutputPath))
        processedImageData = `data:image/jpeg;base64,${processedImageBuffer.toString("base64")}`
      } else {
        throw new Error("Segmentation did not return image data")
      }

      console.log("[Segmentation] Inference done")

      // Step 4: Get final answer from GPT-4V
      console.log("Getting final answer...")
      const answer = await getFinalAnswer(processedImageData, userQuery, refinedQuery, "segmentation")
      console.log("Answer generated:", answer.substring(0, 100) + "...")

      return NextResponse.json({
        processedImageData,
        refinedQuery,
        processingType: "segmentation",
        parameters,
        answer,
        backend: result.backend
      })
    } catch (err: any) {
      console.error("[Segmentation] Model server error:", err?.message || String(err))
      return NextResponse.json({ error: err?.message || "Segmentation model server error" }, { status: 502 })
    }

  } catch (error) {
    console.error("API Error:", error)
    return NextResponse.json({ error: "Error processing request" }, { status: 500 })
  }
} 