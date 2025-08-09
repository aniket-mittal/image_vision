import { NextRequest, NextResponse } from "next/server"

// OpenAI API key
const OPENAI_API_KEY = process.env.OPENAI_API_KEY as string



async function processImageForMode(processingMode: string, imageData: string, userQuery: string) {
  // For now, just return the original image and query
  // The processing will be handled by the frontend before sending to this API
  return { processedImageData: imageData, refinedQuery: userQuery }
}

export async function POST(req: NextRequest) {
  try {
    const { messages, processingMode, aiModel, imageData, selection, useOriginalImage } = await req.json()

    console.log("Received request:", { processingMode, aiModel, useOriginalImage, hasImage: !!imageData })

    let processedImageData = imageData
    let refinedQuery = messages[messages.length - 1]?.content || ""

    // Handle "Use Original Image" case
    if (useOriginalImage && imageData) {
      console.log("Processing original image mode - using user query directly...")
      // Use the original image data and user query without any preprocessing
      processedImageData = imageData
      // Keep the original user query as is
    } else if (!useOriginalImage && imageData) {
      // Process image if not using original image
      try {
        const processingResult = await processImageForMode(processingMode, imageData, refinedQuery)
        processedImageData = processingResult.processedImageData
        refinedQuery = processingResult.refinedQuery
      } catch (error) {
        console.error("Image processing failed:", error)
        // Fallback to original image
        processedImageData = imageData
      }
    }

    // Build system message based on processing mode
    let systemMessage = `You are an AI assistant specialized in image analysis. `

    switch (processingMode) {
      case "Auto":
        systemMessage += `You are analyzing an image that has been automatically processed to highlight the most relevant elements for the user's question. Focus your analysis on the highlighted regions and provide a comprehensive answer.`
        break
      case "Attention":
        systemMessage += `You are analyzing an image with attention masking applied. Focus on the highlighted regions that are most relevant to the user's question.`
        break
      case "Segmentation":
        systemMessage += `You are analyzing an image with segmentation masking applied. Focus on the segmented objects and their boundaries.`
        break
      case "Injection":
        systemMessage += `Pay special attention to user-selected regions and inject additional focus on those areas in your analysis.`
        break
      default:
        systemMessage += `Provide comprehensive analysis of the image, considering all visible elements and their relationships.`
    }

    if (useOriginalImage) {
      systemMessage += ` You are analyzing the original image without any processing applied. Provide a comprehensive analysis based on the user's question.`
    }

    if (selection && !useOriginalImage) {
      systemMessage += ` The user has made a ${selection.type} selection in the image. Pay special attention to the selected region.`
    }

    systemMessage += ` The user has selected the ${aiModel} model for analysis.`

    console.log("System message:", systemMessage)

    // Prepare messages with processed image data
    const enhancedMessages = messages.map((message: any) => {
      if (message.role === "user" && processedImageData && message.content) {
        return {
          role: message.role,
          content: [
            { type: "text", text: message.content },
            { type: "image_url", image_url: { url: processedImageData } },
          ],
        }
      }
      return {
        role: message.role,
        content: message.content
      }
    })

    console.log("Enhanced messages:", enhancedMessages.length)

    try {
      // Use OpenAI API directly
      const openaiResponse = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [
            { role: "system", content: systemMessage },
            ...enhancedMessages
          ],
          max_tokens: 1000,
          stream: true,
        }),
      })

      if (!openaiResponse.ok) {
        throw new Error(`OpenAI API error: ${openaiResponse.statusText}`)
      }

      return new Response(openaiResponse.body, {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      })
    } catch (openaiError) {
      console.error("OpenAI API Error:", openaiError)
      return new Response(JSON.stringify({
        error: "OpenAI API error",
        details: openaiError instanceof Error ? openaiError.message : "Unknown error"
      }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      })
    }

  } catch (error) {
    console.error("API Error:", error)
    return new Response(JSON.stringify({
      error: "General API error",
      details: error instanceof Error ? error.message : "Unknown error"
    }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    })
  }
}
