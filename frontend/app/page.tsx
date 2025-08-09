"use client"

import type React from "react"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Crop, Lasso, MousePointer, Upload, Send, Bot, User, ChevronLeft, ChevronRight, Eraser } from "lucide-react"

type Tool = "crop" | "lasso" | "object-selector"
type ProcessingMode = "Auto" | "Attention" | "Segmentation" | "Injection"
type AIModel = "GPT" | "Claude" | "Gemini" | "LLaVA" | "Qwen"

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
  const [selectedTool, setSelectedTool] = useState<Tool>("crop")
  const [useOriginalImage, setUseOriginalImage] = useState(true)
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("Auto")
  const [aiModel, setAIModel] = useState<AIModel>("GPT")
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [selections, setSelections] = useState<Selection[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawPath, setDrawPath] = useState<number[]>([])
  const [isDetectingObjects, setIsDetectingObjects] = useState(false)

  // Custom chat state
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)

  // Cached object detections from backend: list of { id, bbox: [x1,y1,x2,y2], label }
  const [detectedObjects, setDetectedObjects] = useState<Array<{id:number; bbox:number[]; label:string}>>([])
  const [selectedObjectId, setSelectedObjectId] = useState<number | null>(null)
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

      // 1) Draw blurred copy if any selection exists OR if object-selector is active (pre-blur UX)
      const shouldPreBlur = selectedTool === "object-selector"
      if (paths.length > 0 || shouldPreBlur) {
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
      // First, process the image if not using original
      if (!useOriginalImage && processingMode !== "Injection") {
        try {
          let endpoint = ""
          let requestBody: any = {
            imageData: uploadedImage,
            userQuery: input,
          }

          // Check if any crop or lasso selection exists; use the most recent one for processing
          const roiSelections = selections.filter((s) => s.type === "crop" || s.type === "lasso")
          const lastSelection = roiSelections.length > 0 ? roiSelections[roiSelections.length - 1] : null
          if (lastSelection) {
            endpoint = "/api/crop-lasso-process"
            requestBody = {
              imageData: uploadedImage,
              userQuery: input,
              toolType: lastSelection.type,
              coordinates: lastSelection.coordinates,
            }
          } else {
            switch (processingMode) {
              case "Auto":
                endpoint = "/api/auto-process"
                break
              case "Attention":
                endpoint = "/api/attention-process"
                break
              case "Segmentation":
                endpoint = "/api/segmentation-process"
                break
            }
          }

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
              setProcessedImage(processResult.processedImageData)
              
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
        setProcessedImage(null)
      }

      console.log("Calling chat API with useOriginalImage:", useOriginalImage)
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
          useOriginalImage,
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
        setSelectedObjectId(null)
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
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Choose the first object bbox that contains the click
    // Scale bboxes to display size
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

    if (hit) {
      setSelectedObjectId(hit.id)
    }

    // Map display coordinates to original image coordinates
    let origX = x
    let origY = y
    if (imageRef.current && imageNaturalSize) {
      const sx = imageNaturalSize.w / imageRef.current.offsetWidth
      const sy = imageNaturalSize.h / imageRef.current.offsetHeight
      origX = x * sx
      origY = y * sy
    }

    // Call backend to run SAM-click
    try {
      await runSamClickAt(Math.round(origX), Math.round(origY))
    } catch (_) {}
  }, [selectedTool, detectedObjects, uploadedImage, imageNaturalSize, drawHighlightOverlay])

  // Helper: run SAM-click at original-image coordinates
  const runSamClickAt = useCallback(async (origX: number, origY: number) => {
    if (!uploadedImage) return
    console.log("[ObjectSelector] SAM-click inference start", { x: origX, y: origY })
    const res = await fetch("/api/objects", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageData: uploadedImage, x: origX, y: origY, blurStrength: 15 }),
    })
    if (res.ok) {
      const data = await res.json()
      setProcessedImage(data.processedImageData)
      // Update overlay to match returned bbox (scale to display)
      if (data.bbox && imageNaturalSize && imageRef.current) {
        const [bx1, by1, bx2, by2] = data.bbox as number[]
        const scaleX2 = imageRef.current.offsetWidth / imageNaturalSize.w
        const scaleY2 = imageRef.current.offsetHeight / imageNaturalSize.h
        const sel: Selection = {
          type: "object-selector",
          coordinates: [bx1 * scaleX2, by1 * scaleY2, bx2 * scaleX2, by2 * scaleY2],
          imageData: uploadedImage,
        }
        setSelections([sel])
        drawHighlightOverlay([sel])
        console.log("[ObjectSelector] SAM-click inference done")
      }
    }
  }, [uploadedImage, imageNaturalSize, drawHighlightOverlay])

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
          // Auto-select first object to highlight and pre-blur background
          if ((data.objects || []).length > 0 && imageRef.current && imageNaturalSize) {
            const first = data.objects[0]
            // Compute click point at object center in original image coords
            const [x1, y1, x2, y2] = first.bbox as number[]
            const cx = Math.round((x1 + x2) / 2)
            const cy = Math.round((y1 + y2) / 2)
            await runSamClickAt(cx, cy)
          }
        }
      } catch (_) {}
      finally {
        setIsDetectingObjects(false)
      }
    }
    detect()
  }, [uploadedImage, imageNaturalSize, runSamClickAt])

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
                    setDrawPath([])
                    const canvas = canvasRef.current
                    const ctx = canvas?.getContext("2d")
                    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
                    if (processedImage && processedImage !== uploadedImage) {
                      setProcessedImage(uploadedImage)
                    }
                  }}
                >
                  <Eraser className="w-4 h-4 mr-2" />
                  Erase
                </Button>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <Switch 
                id="use-original" 
                checked={useOriginalImage} 
                onCheckedChange={(checked) => {
                  setUseOriginalImage(checked)
                  if (checked) {
                    setProcessedImage(null)
                  }
                }} 
              />
              <Label htmlFor="use-original">Use Original Image</Label>
            </div>
          </div>
        </div>

        {/* Image Display Area */}
        <div className="flex-1 flex items-center justify-center p-8 relative">
          {/* Detect-all progress bar */}
          {isDetectingObjects && (
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 w-[60%] h-1.5 bg-gray-200 rounded overflow-hidden shadow">
              <div className="h-full bg-blue-500 animate-pulse" style={{ width: "60%" }} />
            </div>
          )}
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
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Processing</Label>
                  <Select value={processingMode} onValueChange={(value: ProcessingMode) => setProcessingMode(value)}>
                    <SelectTrigger className="h-9 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Auto">Auto</SelectItem>
                      <SelectItem value="Attention">Attention</SelectItem>
                      <SelectItem value="Segmentation">Segmentation</SelectItem>
                      <SelectItem value="Injection">Injection</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Model</Label>
                  <Select value={aiModel} onValueChange={(value: AIModel) => setAIModel(value)}>
                    <SelectTrigger className="h-9 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="GPT">GPT-4</SelectItem>
                      <SelectItem value="Claude">Claude</SelectItem>
                      <SelectItem value="Gemini">Gemini</SelectItem>
                      <SelectItem value="LLaVA">LLaVA</SelectItem>
                      <SelectItem value="Qwen">Qwen</SelectItem>
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
