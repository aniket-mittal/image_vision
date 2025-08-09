import { NextRequest, NextResponse } from "next/server"
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

async function refineQueryForAttention(userQuery: string, imageCaption: string): Promise<{
  refinedQuery: string
  parameters: any
}> {
  try {
    console.log("Calling OpenAI for attention refinement...")
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
            content: `You are an expert at refining queries for attention mask generation. Your primary goal is to extract the key objects/concepts from the user's question that need to be highlighted with attention masking.

IMPORTANT: Prioritize the user's question over the image description. The user's question contains the specific objects they want to focus on.

Guidelines:
1. Extract ALL relevant objects mentioned in the user's question
2. If the question involves relationships between objects, include both objects
3. Use simple, clear object names separated by "and" if multiple objects
4. Focus on what the user is actually asking about, not what's in the image description

Examples:
- "What direction are the birds facing in relation to the elephant?" → "birds and elephant"
- "Where is the cat looking?" → "cat"
- "How do the people interact with the building?" → "people and building"
- "What's between the tree and the car?" → "tree and car"

Respond in JSON format:
{
  "refinedQuery": "extracted objects for attention masking",
  "parameters": {
    "layer_index": 23,
    "enhancement_control": 5.0,
    "smoothing_kernel": 3,
    "overlay_strength": 1.0,
    "grayscale_level": 200
  }
}`
          },
          {
            role: "user",
            content: `User Question: \"${userQuery}\"\nImage Description: \"${imageCaption}\"\n\nWhat should be the refined query and parameters for attention masking?`
          }
        ],
        max_tokens: 200
      })
    })

    console.log("OpenAI response status:", response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error("OpenAI API error in refineQueryForAttention:", response.status, errorText)
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`)
    }

    const result = await response.json()
    console.log("OpenAI response result:", JSON.stringify(result, null, 2))
    
    if (!result.choices || !result.choices[0] || !result.choices[0].message) {
      console.error("Unexpected OpenAI response format in refineQueryForAttention:", result)
      throw new Error("Unexpected OpenAI response format")
    }

    const content = result.choices[0].message.content
    console.log("OpenAI response content for attention refinement:", content)
    
    try {
      // Remove markdown code blocks if present
      const cleanContent = content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim()
      const parsed = JSON.parse(cleanContent)
      console.log("Successfully parsed JSON:", parsed)
      return parsed
    } catch (parseError) {
      console.error("Error parsing JSON from OpenAI response in refineQueryForAttention:", parseError)
      console.error("Raw content:", content)
      // Fallback
      return {
        refinedQuery: userQuery,
        parameters: {
          layer_index: 23,
          enhancement_control: 5.0,
          smoothing_kernel: 3,
          overlay_strength: 1.0,
          grayscale_level: 200
        }
      }
    }
  } catch (error) {
    console.error("Error in refineQueryForAttention:", error)
    // Fallback
    return {
      refinedQuery: userQuery,
      parameters: {
        layer_index: 23,
        enhancement_control: 5.0,
        smoothing_kernel: 3,
        overlay_strength: 1.0,
        grayscale_level: 200
      }
    }
  }
}

async function runAttentionMasking(imagePath: string, query: string, parameters: any): Promise<{ savedPath?: string; processedImageData?: string; backend: "model-server" }> {
  // Only call the model server; no local fallbacks
  const requestBody = {
    image_path: imagePath,
    image_data: await import("fs/promises").then(fs => fs.readFile(imagePath).then(b=>`data:image/jpeg;base64,${b.toString("base64")}`)),
    query,
    layer_index: parameters.layer_index ?? 23,
    enhancement_control: parameters.enhancement_control ?? 5.0,
    smoothing_kernel: parameters.smoothing_kernel ?? 3,
    grayscale_level: parameters.grayscale_level ?? 200,
    overlay_strength: parameters.overlay_strength ?? 1.0,
    output_dir: "temp_output",
  }
  const MODEL_SERVER_URL = (process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "")
  const res = await fetch(`${MODEL_SERVER_URL}/attention`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  })
  if (!res.ok) {
    throw new Error(`Model server attention failed ${res.status}: ${await res.text()}`)
  }
  const data = await res.json()
  console.log("[Attention] Using model-server backend")
  return { savedPath: data.saved, processedImageData: data.processedImageData, backend: "model-server" }
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

    console.log("[Attention] Starting")

    // Step 1: Generate image caption
    const imageCaption = await generateImageCaption(imageData)
    console.log("[Attention] Caption done")

    // Step 2: Refine query for attention masking
    const { refinedQuery, parameters } = await refineQueryForAttention(userQuery, imageCaption)
    console.log("[Attention] Refinement done", { refinedQuery, parameters })

    // Step 3: Run attention masking
    const tempDir = join(tmpdir(), "image_vision_temp")
    await mkdir(tempDir, { recursive: true })
    
    const imageBuffer = Buffer.from(imageData.split(",")[1], "base64")
    const imagePath = join(tempDir, "input_image.jpg")
    await writeFile(imagePath, imageBuffer)

    try {
      console.log("[Attention] Inference start")
      const result = await runAttentionMasking(imagePath, refinedQuery, parameters)
      let processedImageData: string
      if (result.processedImageData) {
        processedImageData = result.processedImageData
      } else if (result.savedPath) {
        const absoluteOutputPath = join("/Users/aniketmittal/Desktop/code/image_vision", result.savedPath)
        const processedImageBuffer = await import("fs/promises").then(fs => fs.readFile(absoluteOutputPath))
        processedImageData = `data:image/jpeg;base64,${processedImageBuffer.toString("base64")}`
      } else {
        throw new Error("Model server did not return image data")
      }

      console.log("[Attention] Inference done")

      // Step 4: Get final answer from GPT-4V
      console.log("Getting final answer...")
      const answer = await getFinalAnswer(processedImageData, userQuery, refinedQuery, "attention")
      console.log("Answer generated:", answer.substring(0, 100) + "...")

      return NextResponse.json({
        processedImageData,
        refinedQuery,
        processingType: "attention",
        parameters,
        answer,
        backend: result.backend
      })
    } catch (err: any) {
      console.error("[Attention] Model server error:", err?.message || String(err))
      return NextResponse.json({ error: err?.message || "Attention model server error" }, { status: 502 })
    }

  } catch (error) {
    console.error("API Error:", error)
    return NextResponse.json({ error: "Error processing request" }, { status: 500 })
  }
} 