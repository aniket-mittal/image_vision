import { NextRequest, NextResponse } from "next/server"
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"
import { createCanvas, loadImage } from "canvas"

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
    return "An image with various objects and elements that may be relevant to the user's question."
  }
}

async function createMaskedImage(
  imageData: string, 
  toolType: "crop" | "lasso", 
  coordinates: number[]
): Promise<string> {
  try {
    console.log(`Creating ${toolType} mask with coordinates:`, coordinates)
    
    // Remove data URL prefix to get base64 data
    const base64Data = imageData.replace(/^data:image\/[a-z]+;base64,/, '')
    const imageBuffer = Buffer.from(base64Data, 'base64')
    
    // Load the image
    const img = await loadImage(imageBuffer)
    
    // Create canvas
    const canvas = createCanvas(img.width, img.height)
    const ctx = canvas.getContext('2d')
    
    // Draw the original image
    ctx.drawImage(img, 0, 0)
    
    if (toolType === "crop") {
      // For crop: keep the selected rectangle, blur the rest
      const [x1, y1, x2, y2] = coordinates
      const left = Math.min(x1, x2)
      const top = Math.min(y1, y2)
      const width = Math.abs(x2 - x1)
      const height = Math.abs(y2 - y1)
      
      // Create a temporary canvas for blurring
      const tempCanvas = createCanvas(img.width, img.height)
      const tempCtx = tempCanvas.getContext('2d')
      
      // Draw the original image to temp canvas
      tempCtx.drawImage(img, 0, 0)
      
      // Apply blur effect (simulate blur by drawing with transparency)
      tempCtx.globalAlpha = 0.3
      tempCtx.drawImage(tempCanvas, 0, 0)
      tempCtx.globalAlpha = 0.3
      tempCtx.drawImage(tempCanvas, 0, 0)
      
      // Draw the blurred image to main canvas
      ctx.drawImage(tempCanvas, 0, 0)
      
      // Clear the selected area and draw the original content
      ctx.globalCompositeOperation = 'source-over'
      ctx.drawImage(img, left, top, width, height, left, top, width, height)
      
    } else if (toolType === "lasso") {
      // For lasso: create a closed path and blur everything outside
      if (coordinates.length < 6) {
        throw new Error("Lasso selection needs at least 3 points")
      }
      
      // Create a temporary canvas for blurring
      const tempCanvas = createCanvas(img.width, img.height)
      const tempCtx = tempCanvas.getContext('2d')
      
      // Draw the original image to temp canvas
      tempCtx.drawImage(img, 0, 0)
      
      // Apply blur effect (simulate blur by drawing with transparency)
      tempCtx.globalAlpha = 0.3
      tempCtx.drawImage(tempCanvas, 0, 0)
      tempCtx.globalAlpha = 0.3
      tempCtx.drawImage(tempCanvas, 0, 0)
      
      // Draw the blurred image to main canvas
      ctx.drawImage(tempCanvas, 0, 0)
      
      // Create the lasso path
      ctx.beginPath()
      ctx.moveTo(coordinates[0], coordinates[1])
      for (let i = 2; i < coordinates.length; i += 2) {
        ctx.lineTo(coordinates[i], coordinates[i + 1])
      }
      ctx.closePath()
      
      // Create a clipping path for the lasso selection
      ctx.save()
      ctx.clip()
      
      // Draw the original image content inside the lasso
      ctx.globalCompositeOperation = 'source-over'
      ctx.drawImage(img, 0, 0)
      
      ctx.restore()
    }
    
    // Convert to base64
    const buffer = canvas.toBuffer('image/jpeg', { quality: 0.9 })
    const base64 = buffer.toString('base64')
    return `data:image/jpeg;base64,${base64}`
    
  } catch (error) {
    console.error("Error creating masked image:", error)
    return imageData // Return original image if masking fails
  }
}

async function getFinalAnswer(
  maskedImageData: string, 
  userQuery: string, 
  imageCaption: string, 
  toolType: string
): Promise<string> {
  try {
    console.log("Getting final answer from GPT-4V for crop/lasso...")
    
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

The image has been processed using ${toolType} tool. The user selected a specific region of interest, and the rest of the image has been blurred to focus attention on their selection.

Image description: "${imageCaption}"

Please provide a comprehensive, detailed answer to the user's question based on the masked image. Focus on the clear, unblurred region that the user selected. Be thorough and include all relevant details you can observe in the selected area.

If you cannot see the image clearly or if the image appears to be corrupted, please let the user know.`
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: `Please analyze this ${toolType}-masked image and answer the following question in detail: "${userQuery}"`
              },
              {
                type: "image_url",
                image_url: {
                  url: maskedImageData
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
    const { imageData, userQuery, toolType, coordinates, skipAnswer } = await req.json()

    if (!imageData || !userQuery || !toolType || !coordinates) {
      return NextResponse.json({ 
        error: "Missing required fields: imageData, userQuery, toolType, or coordinates" 
      }, { status: 400 })
    }

    if (!["crop", "lasso"].includes(toolType)) {
      return NextResponse.json({ 
        error: "Invalid toolType. Must be 'crop' or 'lasso'" 
      }, { status: 400 })
    }

    console.log(`Starting ${toolType} processing...`)

    // Step 1: Generate image caption (unless explicitly skipped)
    const imageCaption = skipAnswer ? "" : await generateImageCaption(imageData)
    if (!skipAnswer) console.log("Image caption:", imageCaption)

    // Step 2: Create masked image based on user selection
    console.log("Creating masked image...")
    const maskedImageData = await createMaskedImage(imageData, toolType, coordinates)
    console.log("Masked image created successfully")

    if (!skipAnswer) {
      // Step 3: Get final answer from GPT-4V
      console.log("Getting final answer...")
      const answer = await getFinalAnswer(maskedImageData, userQuery, imageCaption, toolType)
      console.log("Answer generated:", answer.substring(0, 100) + "...")
      return NextResponse.json({
        processedImageData: maskedImageData,
        refinedQuery: `User selected ${toolType} region`,
        processingType: toolType,
        parameters: { coordinates, toolType },
        answer,
      })
    }
    return NextResponse.json({
      processedImageData: maskedImageData,
      refinedQuery: `User selected ${toolType} region`,
      processingType: toolType,
      parameters: { coordinates, toolType },
    })

  } catch (error) {
    console.error("API Error:", error)
    return NextResponse.json({ error: "Error processing request" }, { status: 500 })
  }
} 