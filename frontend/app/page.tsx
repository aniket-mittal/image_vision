"use client"

import type React from "react"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Crop, Lasso, MousePointer, Upload, Send, Bot, User, ChevronLeft, ChevronRight, Eraser } from "lucide-react"

type Tool = "crop" | "lasso" | "object-selector"
type Mode = "Ask" | "Edit"
type ProcessingMode = "Original" | "BlurLight" | "BlurDeep" | "Injection"
type AIModel =
  | "GPT"
  | "Claude"
  | "Gemini"
  | "LLaVA"
  | "Qwen"
  | "Auto"
  | "OpenAIImages"
  | "SDXL"
  | "StabilityAI"
  | "LaMa"

interface Selection {
  type: Tool
  coordinates: number[]
  imageData?: string | null
}

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
}

export default function AIImageAnalyzer() {
  const [mode, setMode] = useState<Mode>("Ask")
  const [selectedTool, setSelectedTool] = useState<Tool>("crop")
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("Original")
  const [aiModel, setAIModel] = useState<AIModel>("GPT")
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [selections, setSelections] = useState<Selection[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawPath, setDrawPath] = useState<number[]>([])
  const [isDetectingObjects, setIsDetectingObjects] = useState(false)
  const [hasAutoHighlighted, setHasAutoHighlighted] = useState(false)

  // Custom chat state
  const [messages, setMessages] = useState<Message[]>([])
  const [agentEvents, setAgentEvents] = useState<string[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)

  // Cached object detections from backend: list of { id, bbox: [x1,y1,x2,y2], label }
  const [detectedObjects, setDetectedObjects] = useState<Array<{id:number; bbox:number[]; label:string}>>([])
  const [selectedObjectIds, setSelectedObjectIds] = useState<number[]>([])
  const [selectedObjectPoints, setSelectedObjectPoints] = useState<Array<{id:number; x:number; y:number}>>([])
  const [imageNaturalSize, setImageNaturalSize] = useState<{w:number; h:number} | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Draw blurred background with highlighted selection (crop or lasso)
  const drawHighlightOverlay = useCallback(
    (
      committedSelections: Selection[] | null,
      pending?: { coordinates: number[]; type: Tool } | null,
    ) => {
      const canvas = canvasRef.current
      const imgEl = imageRef.current
      if (!canvas || !imgEl) return
      const ctx = canvas.getContext("2d")
      if (!ctx) return

      // Clear previous overlay
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const paths: { type: Tool; coordinates: number[] }[] = []
      if (committedSelections && committedSelections.length > 0) {
        for (const sel of committedSelections) {
          if (sel.type === "crop" || sel.type === "lasso" || sel.type === "object-selector") {
            paths.push({ type: sel.type, coordinates: sel.coordinates })
          }
        }
      }
      if (pending && (pending.type === "crop" || pending.type === "lasso" || pending.type === "object-selector") && pending.coordinates.length > 0) {
        paths.push(pending)
      }
      if (paths.length === 0) return

      const drawOnePath = (p: { type: Tool; coordinates: number[] }) => {
        ctx.beginPath()
        if ((p.type === "crop" || p.type === "object-selector") && p.coordinates.length === 4) {
          const [x1, y1, x2, y2] = p.coordinates
          const left = Math.min(x1, x2)
          const top = Math.min(y1, y2)
          const width = Math.abs(x2 - x1)
          const height = Math.abs(y2 - y1)
          ctx.rect(left, top, width, height)
        } else if (p.type === "lasso" && p.coordinates.length > 4) {
          ctx.moveTo(p.coordinates[0], p.coordinates[1])
          for (let i = 2; i < p.coordinates.length; i += 2) {
            ctx.lineTo(p.coordinates[i], p.coordinates[i + 1])
          }
          ctx.closePath()
        }
      }

      // 1) Draw blurred copy only if there is a selection to highlight
      const shouldBlur = paths.length > 0
      if (shouldBlur) {
        ctx.save()
        ctx.filter = "blur(8px)"
        ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height)
        ctx.restore()
      }

      // 2) Slightly darken to emphasize focus
      ctx.save()
      ctx.fillStyle = "rgba(0,0,0,0.25)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.restore()

      // 3) Punch out the selection area to reveal the sharp image underneath
      if (paths.length > 0) {
        ctx.save()
        ctx.globalCompositeOperation = "destination-out"
        for (const p of paths) {
          drawOnePath(p)
          ctx.fillStyle = "rgba(255,255,255,1)"
          ctx.fill()
        }
        ctx.restore()
      }

      // 4) Draw a highlight stroke around the selection
      if (paths.length > 0) {
        ctx.save()
        ctx.strokeStyle = "#3b82f6"
        ctx.lineWidth = 2
        ctx.shadowColor = "rgba(59,130,246,0.5)"
        ctx.shadowBlur = 4
        for (const p of paths) {
          drawOnePath(p)
          ctx.stroke()
        }
        ctx.restore()
      }

      // 5) If object-selector is active, draw detected object boxes as guidance
      if (selectedTool === "object-selector" && detectedObjects.length > 0 && imageNaturalSize) {
        const scaleX = canvas.width / imageNaturalSize.w
        const scaleY = canvas.height / imageNaturalSize.h
        ctx.save()
        ctx.strokeStyle = "rgba(255,255,255,0.6)"
        ctx.lineWidth = 1
        detectedObjects.forEach((o) => {
          const [bx1, by1, bx2, by2] = o.bbox
          const x1 = bx1 * scaleX
          const y1 = by1 * scaleY
          const w = (bx2 - bx1) * scaleX
          const h = (by2 - by1) * scaleY
          ctx.strokeRect(x1, y1, w, h)
        })
        ctx.restore()
      }
    },
    [selectedTool, detectedObjects, imageNaturalSize],
  )

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !uploadedImage || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // First, process the image if processing is requested in Ask mode (not Original or Injection)
      if (mode === "Ask" && processingMode !== "Original" && processingMode !== "Injection") {
        try {
          // Use agentic orchestrator for Blur modes
          let endpoint = "/api/ask-agent"
          const roiSelections = selections.filter((s) => s.type === "crop" || s.type === "lasso")
          const lastSelection = roiSelections.length > 0 ? roiSelections[roiSelections.length - 1] : null
          let seedImageData: string | null = null
          if (lastSelection) {
            // Create a quick masked seed via existing API (without using its answer)
            try {
              const seedResp = await fetch("/api/crop-lasso-process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  imageData: uploadedImage,
                  userQuery: input,
                  toolType: lastSelection.type,
                  coordinates: lastSelection.coordinates,
                  skipAnswer: true,
                }),
              })
              if (seedResp.ok) {
                const seedJson = await seedResp.json()
                seedImageData = seedJson.processedImageData || null
              }
            } catch (e) {
              // ignore seed errors; agent can still proceed
            }
          }
          let requestBody: any = {
            imageData: uploadedImage,
            userQuery: input,
            mode: processingMode === "BlurDeep" ? "BlurDeep" : "BlurLight",
            aiModel,
            selection: lastSelection
              ? { type: lastSelection.type, coordinates: lastSelection.coordinates }
              : null,
            seedImageData,
          }

          // If lastSelection exists, we can create a quick masked seed preview on the client canvas as a seed for the agent
          // but for now we just pass selection metadata. The agent will handle ROI-aware planning.

          if (endpoint) {
            const processResponse = await fetch(endpoint, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify(requestBody),
            })

            if (processResponse.ok) {
              const processResult = await processResponse.json()
              console.log("Processing result:", processResult)
              const finalImg = processResult?.final?.processedImageData || processResult.processedImageData
              if (finalImg) setProcessedImage(finalImg)
              // Append chain-of-thought style telemetry for the UI (not revealing model internals)
              if (processResult?.steps && Array.isArray(processResult.steps)) {
                const logs: string[] = []
                processResult.steps.forEach((s: any, idx: number) => {
                  const parts = [] as string[]
                  parts.push(`Step ${idx + 1}: ${s.technique} → "${s.refinedQuery}"`)
                  if (s.params) parts.push(`params=${JSON.stringify(s.params)}`)
                  if (s.rationale) parts.push(`note=${s.rationale}`)
                  logs.push(parts.join(" | "))
                })
                setAgentEvents((prev) => [...prev, ...logs])
              }
              
              // If the processing API returned an answer, use it directly
              if (processResult.answer) {
                console.log("Using answer from processing API:", processResult.answer.substring(0, 100) + "...")
                setMessages((prev) => [
                  ...prev,
                  {
                    id: (Date.now() + 1).toString(),
                    role: "assistant",
                    content: processResult.answer,
                  },
                ])
                setIsLoading(false)
                return // Skip the chat API call since we already have the answer
              } else {
                console.log("No answer from processing API, proceeding to chat API")
              }
            } else {
              console.error("Processing API failed:", processResponse.status, await processResponse.text())
            }
          }
        } catch (error) {
          console.error("Image processing failed:", error)
          setProcessedImage(null)
        }
      } else {
        // For Original/Injection, or in Edit mode, ensure we send the original image to chat
        setProcessedImage(null)
        // Also clear any overlays to prevent perceived blur over the original
        const canvas = canvasRef.current
        const ctx = canvas?.getContext("2d")
        if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
      }

      console.log("Calling chat API (processedImage present?):", !!processedImage)
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
          body: JSON.stringify({
          messages: [...messages, userMessage],
          processingMode,
          aiModel,
          imageData: uploadedImage,
          processedImageData: processedImage,
            // For chat context, include the most recent ROI selection if any
            selection: (selections.filter((s) => s.type === "crop" || s.type === "lasso").slice(-1)[0] ?? null),
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response")
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error("No response body")
      }

      let assistantMessage = ""
      const assistantMessageId = (Date.now() + 1).toString()

      // Add empty assistant message that we'll update
      setMessages((prev) => [
        ...prev,
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
        },
      ])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = new TextDecoder().decode(value)
        const lines = chunk.split("\n")

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.choices && data.choices[0] && data.choices[0].delta && data.choices[0].delta.content) {
                assistantMessage += data.choices[0].delta.content
                setMessages((prev) =>
                  prev.map((msg) => (msg.id === assistantMessageId ? { ...msg, content: assistantMessage } : msg)),
                )
              }
            } catch (e) {
              // Ignore parsing errors
            }
          }
        }
        
        console.log("Processed chunk, current assistant message length:", assistantMessage.length)
      }
    } catch (error) {
      console.error("Error:", error)
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Sorry, I encountered an error while processing your request.",
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
        setSelections([])
        setDetectedObjects([])
        setSelectedObjectIds([])
        setSelectedObjectPoints([])
        setHasAutoHighlighted(false)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const handleCanvasMouseDown = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current) return

      const rect = canvasRef.current.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top

      setIsDrawing(true)

      if (selectedTool === "lasso") {
        setDrawPath([x, y])
      } else if (selectedTool === "crop") {
        setDrawPath([x, y, x, y])
      }
    },
    [selectedTool],
  )

  const handleCanvasMouseMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDrawing || !canvasRef.current) return

      const rect = canvasRef.current.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top

      if (selectedTool === "lasso") {
        setDrawPath((prev) => {
          const next = [...prev, x, y]
          drawHighlightOverlay(selections, { coordinates: next, type: "lasso" })
          return next
        })
      } else if (selectedTool === "crop") {
        setDrawPath((prev) => {
          const next = [prev[0], prev[1], x, y]
          drawHighlightOverlay(selections, { coordinates: next, type: "crop" })
          return next
        })
      }
    },
    [isDrawing, selectedTool, drawHighlightOverlay, selections],
  )

  const handleCanvasMouseUp = useCallback(() => {
    if (!isDrawing) return

    setIsDrawing(false)

    if (drawPath.length > 0) {
      const selectionData: Selection = {
        type: selectedTool,
        coordinates: drawPath,
        imageData: uploadedImage,
      }
      setSelections((prev) => {
        const next = [...prev, selectionData]
        // Persist overlay after selection finalized
        drawHighlightOverlay(next)
        return next
      })
      setDrawPath([])
    }
  }, [isDrawing, drawPath, selectedTool, uploadedImage, drawHighlightOverlay])

  const handleObjectSelect = useCallback(async (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedTool !== "object-selector") return
    if (!canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Choose the first object bbox that contains the click (display coords)
    const imgEl = imageRef.current
    let scaleX = 1, scaleY = 1
    if (imgEl && imageNaturalSize) {
      scaleX = imgEl.offsetWidth / imageNaturalSize.w
      scaleY = imgEl.offsetHeight / imageNaturalSize.h
    }
    const hit = detectedObjects.find((o) => {
      const [bx1, by1, bx2, by2] = o.bbox
      const x1 = bx1 * scaleX
      const y1 = by1 * scaleY
      const x2 = bx2 * scaleX
      const y2 = by2 * scaleY
      return x >= x1 && x <= x2 && y >= y1 && y <= y2
    })
    if (!hit) return

    // Map display coordinates to original image coordinates
    let origX = x
    let origY = y
    if (imageRef.current && imageNaturalSize) {
      const sx = imageNaturalSize.w / imageRef.current.offsetWidth
      const sy = imageNaturalSize.h / imageRef.current.offsetHeight
      origX = x * sx
      origY = y * sy
    }

    // Toggle selection and recompute
    setSelectedObjectIds((prev) => {
      const exists = prev.includes(hit.id)
      const nextIds = exists ? prev.filter((id) => id !== hit.id) : [...prev, hit.id]

      setSelectedObjectPoints((prevPts) => {
        const map = new Map(prevPts.map((p) => [p.id, p]))
        if (exists) {
          map.delete(hit.id)
        } else {
          map.set(hit.id, { id: hit.id, x: Math.round(origX), y: Math.round(origY) })
        }
        const nextPts = Array.from(map.values())
        // Backend inference for all selected points
        runSamMultiClick(nextPts.map((p) => ({ x: p.x, y: p.y }))).catch(() => {})

        // Update overlay selections with all selected object boxes
        const nonObjectSelections = selections.filter((s) => s.type !== "object-selector")
        let sX = 1, sY = 1
        if (imageRef.current && imageNaturalSize) {
          sX = imageRef.current.offsetWidth / imageNaturalSize.w
          sY = imageRef.current.offsetHeight / imageNaturalSize.h
        }
        const objectSelections = nextIds
          .map((id) => detectedObjects.find((o) => o.id === id))
          .filter(Boolean)
          .map((o: any) => {
            const [bx1, by1, bx2, by2] = o.bbox as number[]
            const coords = [bx1 * sX, by1 * sY, bx2 * sX, by2 * sY]
            return { type: "object-selector" as Tool, coordinates: coords, imageData: uploadedImage }
          })
        const merged = [...nonObjectSelections, ...objectSelections]
        setSelections(merged)
        drawHighlightOverlay(merged)

        return nextPts
      })

      return nextIds
    })
  }, [selectedTool, detectedObjects, uploadedImage, imageNaturalSize, drawHighlightOverlay, selections])

  // Helper: run SAM with multiple click points (original-image coordinates)
  const runSamMultiClick = useCallback(async (points: Array<{ x: number; y: number }>) => {
    if (!uploadedImage) return
    if (!points || points.length === 0) {
      setProcessedImage(null)
      return
    }
    console.log("[ObjectSelector] SAM-multi-click start", { count: points.length })
    const res = await fetch("/api/objects", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageData: uploadedImage, points, blurStrength: 15 }),
    })
    if (!res.ok) return
    const data = await res.json()
    if (!data.processedImageData) return
    setProcessedImage(data.processedImageData)
    console.log("[ObjectSelector] SAM-multi-click done")
  }, [uploadedImage])

  // Clear overlay when selection is cleared or image changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    if (!selections || selections.length === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    } else {
      drawHighlightOverlay(selections)
    }
  }, [selections, uploadedImage, drawHighlightOverlay])

  // Warmup models on first mount
  useEffect(() => {
    fetch("/api/warmup", { method: "POST" }).catch(() => {})
  }, [])

  // Ensure model selection is valid when switching modes
  useEffect(() => {
    const askModels: AIModel[] = ["GPT", "Claude", "Gemini", "LLaVA", "Qwen", "Auto"]
    const editModels: AIModel[] = ["OpenAIImages", "SDXL", "StabilityAI", "LaMa"]
    if (mode === "Ask") {
      if (!askModels.includes(aiModel)) {
        setAIModel("GPT")
      }
    } else {
      if (!editModels.includes(aiModel)) {
        setAIModel("OpenAIImages")
      }
    }
  }, [mode])

  // When an image is uploaded, detect all objects and cache them
  useEffect(() => {
    const detect = async () => {
      if (!uploadedImage) return
      try {
        setIsDetectingObjects(true)
        const res = await fetch("/api/objects", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ imageData: uploadedImage, prompt: "object" }),
        })
        if (res.ok) {
          const data = await res.json()
          setDetectedObjects(data.objects || [])
          // Auto-highlight first object once per image
          if (!hasAutoHighlighted && (data.objects || []).length > 0 && imageRef.current && imageNaturalSize) {
            const first = data.objects[0]
            const [x1, y1, x2, y2] = first.bbox as number[]
            const cx = Math.round((x1 + x2) / 2)
            const cy = Math.round((y1 + y2) / 2)
            setSelectedObjectIds([first.id])
            setSelectedObjectPoints([{ id: first.id, x: cx, y: cy }])
            // Build overlay selection box scaled to canvas
            const sX = imageRef.current.offsetWidth / imageNaturalSize.w
            const sY = imageRef.current.offsetHeight / imageNaturalSize.h
            const sel: Selection = { type: "object-selector", coordinates: [x1 * sX, y1 * sY, x2 * sX, y2 * sY], imageData: uploadedImage }
            const merged = [...selections.filter((s) => s.type !== "object-selector"), sel]
            setSelections(merged)
            drawHighlightOverlay(merged)
            // Fire and forget single-point multi-click
            runSamMultiClick([{ x: cx, y: cy }]).catch(() => {})
            setHasAutoHighlighted(true)
          }
        }
      } catch (_) {}
      finally {
        setIsDetectingObjects(false)
      }
    }
    detect()
  }, [uploadedImage])

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Main Image Area */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant={selectedTool === "crop" ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  setSelectedTool("crop")
                  setDrawPath([])
                }}
              >
                <Crop className="w-4 h-4 mr-2" />
                Crop
              </Button>
              <Button
                variant={selectedTool === "lasso" ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  setSelectedTool("lasso")
                  setDrawPath([])
                }}
              >
                <Lasso className="w-4 h-4 mr-2" />
                Lasso
              </Button>
              <Button
                variant={selectedTool === "object-selector" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedTool("object-selector")}
              >
                <MousePointer className="w-4 h-4 mr-2" />
                Object Selector
              </Button>
              {selections.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSelections([])
                    setSelectedObjectIds([])
                    setSelectedObjectPoints([])
                    setDrawPath([])
                    const canvas = canvasRef.current
                    const ctx = canvas?.getContext("2d")
                    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
                    // ensure original image is shown (exact uploaded quality)
                    setProcessedImage(null)
                  }}
                >
                  <Eraser className="w-4 h-4 mr-2" />
                  Erase
                </Button>
              )}
            </div>

            <div className="flex items-center space-x-2"></div>
          </div>
        </div>

        {/* Image Display Area */}
        <div className="flex-1 flex items-center justify-center p-8 relative">
          {/* Detect-all progress bar removed per request */}
          {uploadedImage ? (
            <div className="relative max-w-full max-h-full">
              <img
                ref={imageRef}
                src={processedImage || uploadedImage || "/placeholder.svg"}
                alt="Uploaded"
                className="max-w-full max-h-full object-contain"
                onLoad={() => {
                  if (canvasRef.current && imageRef.current) {
                    const canvas = canvasRef.current
                    const img = imageRef.current
                    setImageNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })
                    canvas.width = img.offsetWidth
                    canvas.height = img.offsetHeight
                    canvas.style.width = `${img.offsetWidth}px`
                    canvas.style.height = `${img.offsetHeight}px`
                    // Redraw overlay, if there are existing selections
                    if (selections && selections.length > 0) {
                      drawHighlightOverlay(selections)
                    } else {
                      const ctx = canvas.getContext("2d")
                      ctx?.clearRect(0, 0, canvas.width, canvas.height)
                      // If Object Selector is active, still show pre-blur and guide boxes
                      if (selectedTool === "object-selector") {
                        drawHighlightOverlay([])
                      }
                    }
                  }
                }}
              />
              <canvas
                ref={canvasRef}
                className={`absolute top-0 left-0 ${
                  selectedTool === "crop" || selectedTool === "lasso" 
                    ? "cursor-crosshair" 
                    : selectedTool === "object-selector" 
                    ? "cursor-pointer" 
                    : "cursor-default"
                }`}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onClick={selectedTool === "object-selector" ? handleObjectSelect : undefined}
              />
              
              {/* Tool Instructions Overlay */}
              {(selectedTool === "crop" || selectedTool === "lasso") && selections.length === 0 && (
                <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-3 py-2 rounded-lg text-sm">
                  {selectedTool === "crop" ? "Click and drag to select a rectangular region" : "Click and drag to draw a selection path"}
                </div>
              )}
            </div>
          ) : (
            <div className="text-center">
              <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <p className="text-gray-600 mb-4">Upload an image to get started</p>
              <Button onClick={() => fileInputRef.current?.click()}>Choose Image</Button>
            </div>
          )}

          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
        </div>
      </div>

      {/* Chat Sidebar */}
      <div
        className={`bg-white border-l flex flex-col transition-all duration-300 ${isSidebarCollapsed ? "w-12" : "w-96"}`}
      >
        {/* Collapse Toggle */}
        <div className="p-3 border-b flex items-center justify-between">
          {!isSidebarCollapsed && (
            <div className="flex items-center">
              <Bot className="w-5 h-5 mr-2 text-blue-600" />
              <span className="font-semibold text-gray-900">AI Analysis</span>
            </div>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            className="h-8 w-8 p-0"
          >
            {isSidebarCollapsed ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </Button>
        </div>

        {!isSidebarCollapsed && (
          <div className="flex-1 flex flex-col p-4 space-y-4">
            {/* Model Configuration */}
            <div className="space-y-3">
              <div className={`grid ${mode === "Ask" ? "grid-cols-3" : "grid-cols-2"} gap-3`}>
                {/* Mode */}
                <div className="space-y-1">
                  <Label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Mode</Label>
                  <Select value={mode} onValueChange={(value: Mode) => setMode(value)}>
                    <SelectTrigger className="h-9 text-sm w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Ask">Ask</SelectItem>
                      <SelectItem value="Edit">Edit</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Processing (Ask mode only) */}
                {mode === "Ask" && (
                  <div className="space-y-1">
                    <Label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Processing</Label>
                    <Select value={processingMode} onValueChange={(value: ProcessingMode) => setProcessingMode(value)}>
                      <SelectTrigger className="h-9 text-sm w-full">
                      <SelectValue />
                    </SelectTrigger>
                       <SelectContent align="end">
                         <SelectItem value="Original">Original Image</SelectItem>
                         <SelectItem value="BlurLight">Blur - Light</SelectItem>
                         <SelectItem value="BlurDeep">Blur - Deep</SelectItem>
                         <SelectItem value="Injection">Injection</SelectItem>
                       </SelectContent>
                  </Select>
                  </div>
                )}

                {/* Model (Ask or Edit) */}
                <div className="space-y-1">
                  <Label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Model</Label>
                  <Select value={aiModel} onValueChange={(value: AIModel) => setAIModel(value)}>
                    <SelectTrigger className="h-9 text-sm w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {mode === "Ask" ? (
                        <>
                          <SelectItem value="GPT">GPT-4</SelectItem>
                          <SelectItem value="Claude">Claude</SelectItem>
                          <SelectItem value="Gemini">Gemini</SelectItem>
                          <SelectItem value="LLaVA">LLaVA</SelectItem>
                          <SelectItem value="Qwen">Qwen</SelectItem>
                          <SelectItem value="Auto">Auto</SelectItem>
                        </>
                      ) : (
                        <>
                          <SelectItem value="OpenAIImages">OpenAI Images</SelectItem>
                          <SelectItem value="SDXL">SDXL</SelectItem>
                          <SelectItem value="StabilityAI">Stability AI</SelectItem>
                          <SelectItem value="LaMa">LaMa</SelectItem>
                        </>
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Selections Info */}
              {selections.length > 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 space-y-2">
                  {selections.map((s, idx) => (
                    <div key={idx} className="text-xs text-blue-700 flex items-center justify-between">
                      <div>
                        <span className="uppercase tracking-wide font-medium mr-2">{s.type.replace("-", " ")}</span>
                        {s.type === "crop" ? (
                          <span className="text-blue-600">
                            {(() => {
                              const [x1, y1, x2, y2] = s.coordinates
                              return `Region: ${[x1, y1, x2, y2].map((n) => Math.round(n)).join(", ")}`
                            })()}
                          </span>
                        ) : (
                          <span className="text-blue-600">Points: {Math.floor(s.coordinates.length / 2)}</span>
                        )}
                      </div>
                    </div>
                  ))}
                  <div className="text-xs text-blue-500">✓ Ready to analyze selected regions</div>
                </div>
              )}
            </div>

            <Separator />

            {/* Chat Messages */}
            <ScrollArea className="flex-1 -mx-1 px-1">
              <div className="space-y-3">
                {messages.length === 0 && (
                  <div className="text-center py-12">
                    <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
                      <Bot className="w-6 h-6 text-gray-400" />
                    </div>
                    <p className="text-sm text-gray-500 mb-1">Ready to analyze</p>
                    <p className="text-xs text-gray-400">Upload an image and ask questions</p>
                  </div>
                )}

                {messages.map((message) => (
                  <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`flex items-start space-x-2 max-w-[85%] ${
                        message.role === "user" ? "flex-row-reverse space-x-reverse" : ""
                      }`}
                    >
                      <div
                        className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                          message.role === "user" ? "bg-blue-500" : "bg-gray-600"
                        }`}
                      >
                        {message.role === "user" ? (
                          <User className="w-3.5 h-3.5 text-white" />
                        ) : (
                          <Bot className="w-3.5 h-3.5 text-white" />
                        )}
                      </div>
                      <div
                        className={`rounded-2xl px-3 py-2 ${
                          message.role === "user" ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-900 border"
                        }`}
                      >
                        <p className="text-sm leading-relaxed">{message.content}</p>
                      </div>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex justify-start">
                    <div className="flex items-start space-x-2">
                      <div className="w-7 h-7 rounded-full bg-gray-600 flex items-center justify-center">
                        <Bot className="w-3.5 h-3.5 text-white" />
                      </div>
                      <div className="bg-gray-100 border rounded-2xl px-3 py-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div
                            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                            style={{ animationDelay: "0.1s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* Agent activity trace */}
            {agentEvents.length > 0 && (
              <div className="mt-2 border rounded-lg p-2 bg-white">
                <div className="text-xs font-semibold text-gray-700 mb-1">Agent activity</div>
                <div className="space-y-1 max-h-40 overflow-auto text-[11px] text-gray-700">
                  {agentEvents.map((e, i) => (
                    <div key={i} className="flex items-start space-x-1">
                      <span className="text-gray-400">•</span>
                      <span>{e}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Chat Input */}
            <div className="border-t pt-4">
              <form onSubmit={handleSubmit} className="flex space-x-2">
                <div className="flex-1 relative">
                  <Input
                    value={input}
                    onChange={handleInputChange}
                    placeholder="Ask about the image..."
                    disabled={!uploadedImage || isLoading}
                    className="pr-10 h-10 text-sm"
                  />
                </div>
                <Button
                  type="submit"
                  disabled={!uploadedImage || isLoading || !input.trim()}
                  size="sm"
                  className="h-10 w-10 p-0"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
