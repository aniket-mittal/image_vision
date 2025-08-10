import { NextRequest } from "next/server"
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"
import { callChatJSON } from "@/lib/llm"

const MODEL_SERVER_URL = (process.env.MODEL_SERVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "")

type DepthMode = "BlurLight" | "BlurDeep"

interface AgentInput {
  imageData: string
  userQuery: string
  mode: DepthMode
  aiModel?: string
  selection?: { type: string; coordinates: number[] } | null
  seedImageData?: string | null
}

async function callOpenAI(messages: any[], maxTokens = 700) {
  return callChatJSON(messages, { modelHint: "GPT", openAIModel: "gpt-4o", maxTokens })
}

async function runAttention(imagePath: string, query: string, p: any) {
  const body = {
    image_data: await import("fs/promises").then((fs) =>
      fs.readFile(imagePath).then((b) => `data:image/jpeg;base64,${b.toString("base64")}`),
    ),
    query,
    layer_index: p.layer_index ?? 23,
    enhancement_control: p.enhancement_control ?? 5.0,
    smoothing_kernel: p.smoothing_kernel ?? 3,
    grayscale_level: p.grayscale_level ?? 200,
    overlay_strength: p.overlay_strength ?? 1.0,
    spatial_bias: p.spatial_bias,
    bbox_bias: p.bbox_bias,
    bias_strength: p.bias_strength ?? 1.3,
    output_dir: "temp_output",
  }
  const r = await fetch(`${MODEL_SERVER_URL}/attention`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`attention ${r.status}: ${await r.text()}`)
  return r.json() as Promise<{ processedImageData: string; attention_bbox?: number[] }>
}

async function runSegmentation(imagePath: string, query: string, p: any) {
  const body = {
    image_data: await import("fs/promises").then((fs) =>
      fs.readFile(imagePath).then((b) => `data:image/jpeg;base64,${b.toString("base64")}`),
    ),
    query,
    blur_strength: p.blur_strength ?? 15,
    padding: p.padding ?? 20,
    mask_type: p.mask_type || "precise",
    output_dir: "temp_output",
  }
  const r = await fetch(`${MODEL_SERVER_URL}/segmentation`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`segmentation ${r.status}: ${await r.text()}`)
  return r.json() as Promise<{ processedImageData: string; objects?: Array<{ bbox: number[]; label: string }> }>
}

async function runOCR(imagePath: string) {
  const body = {
    image_data: await import("fs/promises").then((fs) =>
      fs.readFile(imagePath).then((b) => `data:image/jpeg;base64,${b.toString("base64")}`),
    ),
  }
  const r = await fetch(`${MODEL_SERVER_URL}/ocr`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`ocr ${r.status}: ${await r.text()}`)
  return r.json()
}

async function runOCRFocus(imagePath: string, region: "title" | "x_axis" | "y_axis" | "legend", blur_strength = 15) {
  const body = {
    image_data: await import("fs/promises").then((fs) =>
      fs.readFile(imagePath).then((b) => `data:image/jpeg;base64,${b.toString("base64")}`),
    ),
    region,
    blur_strength,
  }
  const r = await fetch(`${MODEL_SERVER_URL}/ocr_focus`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`ocr_focus ${r.status}: ${await r.text()}`)
  return r.json() as Promise<{ processedImageData: string; layout: any; region: string }>
}

async function runBlurBox(imagePath: string, x1: number, y1: number, x2: number, y2: number, blur_strength = 15) {
  const body = {
    image_data: await import("fs/promises").then((fs) =>
      fs.readFile(imagePath).then((b) => `data:image/jpeg;base64,${b.toString("base64")}`),
    ),
    x1, y1, x2, y2, blur_strength,
  }
  const r = await fetch(`${MODEL_SERVER_URL}/blur_box`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`blur_box ${r.status}: ${await r.text()}`)
  return r.json() as Promise<{ processedImageData: string; bbox: number[] }>
}

async function runBlurOval(imagePath: string, x1: number, y1: number, x2: number, y2: number, blur_strength = 15) {
  const body = {
    image_data: await import("fs/promises").then((fs) =>
      fs.readFile(imagePath).then((b) => `data:image/jpeg;base64,${b.toString("base64")}`),
    ),
    x1, y1, x2, y2, blur_strength,
  }
  const r = await fetch(`${MODEL_SERVER_URL}/blur_oval`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`blur_oval ${r.status}: ${await r.text()}`)
  return r.json() as Promise<{ processedImageData: string; bbox: number[] }>
}

function maxIterations(mode: DepthMode) {
  return mode === "BlurDeep" ? 20 : 5
}

export async function POST(req: NextRequest) {
  const { imageData, userQuery, mode, selection, seedImageData } = (await req.json()) as AgentInput
  const tempDir = join(tmpdir(), "image_vision_agent")
  await mkdir(tempDir, { recursive: true })
  const imageBuffer = Buffer.from(imageData.split(",")[1], "base64")
  const imagePath = join(tempDir, "agent_input.jpg")
  await writeFile(imagePath, imageBuffer)

  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start: async (controller) => {
      const send = (obj: any) => controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"))
      try {
        // OCR
        let ocr: any = {}
        try { ocr = await runOCR(imagePath) } catch {}
        const ocrSummary = (() => {
          try {
            const c = ocr?.layout?.candidates || {}
            const title = c?.title ? "yes" : "no"
            const xt = Array.isArray(c?.x_axis_ticks) ? c.x_axis_ticks.length : 0
            const yt = Array.isArray(c?.y_axis_ticks) ? c.y_axis_ticks.length : 0
            const lg = Array.isArray(c?.legend_candidates) ? c.legend_candidates.length : 0
            return `layout -> title:${title}, x_ticks:${xt}, y_ticks:${yt}, legend_boxes:${lg}`
          } catch { return "layout -> unknown" }
        })()
        send({ type: "ocr", text: ocr?.text || "", summary: ocrSummary })

        const iters = maxIterations(mode)
        let lastProcessed = imageData
        let lastTechnique = ""
        let lastParams = {}
        let lastSuggestion: string | null = null
        let findings: string[] = []
        let techniqueContext: { [key: string]: string[] } = {}
        let coordinateHistory: { [key: string]: any[] } = {}
        let objectTargets: string[] = []
        
        // Query decomposition: break down user query into sub-questions for comprehensive analysis
        let subQuestions: string[] = []
        try {
          const decompositionPrompt = [
            { role: "system", content: "Break down the user's question into 2-4 specific sub-questions that can be answered by analyzing different aspects of the image. Also identify specific objects, entities, or areas that should be located and analyzed. Return JSON {\"sub_questions\":[\"question1\",\"question2\"],\"target_objects\":[\"object1\",\"object2\"],\"spatial_focus\":\"description of where to look\"}. Respond ONLY with a valid JSON object. Do not include any explanation or code fences." },
            { role: "user", content: `Question: ${userQuery}\n\nBreak this down into specific sub-questions and identify what objects/entities need to be located.` }
          ]
          const decompResp = await callChatJSON(decompositionPrompt)
          subQuestions = decompResp.sub_questions || []
          objectTargets = decompResp.target_objects || []
          if (decompResp.spatial_focus) {
            lastSuggestion = `spatial_focus:${decompResp.spatial_focus}`
          }
        } catch {}
        
        // Limit sub-questions based on mode (Light: ≤2, Deep: ≤5)
        const maxSubQuestions = mode === "BlurLight" ? 2 : 5
        subQuestions = subQuestions.slice(0, maxSubQuestions)

        if (seedImageData) {
          lastProcessed = seedImageData
          send({ type: "image", step: 0, technique: lastTechnique, refinedQuery: userQuery, processedImageData: seedImageData, narrative: "Using user-provided region as initial focus." })
        }

        for (let i = seedImageData ? 1 : 0; i < iters; i++) {
          // Plan next step
          const planPrompt = [
            { role: "system", content: `You are an expert vision analysis planner. Your goal is to intelligently select the best technique for each analysis step, considering:

1. **Verifier Feedback Priority:**
   - ALWAYS try to prefer the verifier's suggestions when making decisions
   - If the verifier suggests using coordinates, consider blur_oval or blur_box
   - If the verifier suggests expanding focus, consider coordinate-based techniques
   - The verifier's feedback provides valuable guidance for your decisions

2. **Technique Selection Strategy:**
   - **Attention**: Best for highlighting specific objects/regions when you know what to look for
   - **Segmentation**: Best for precise object boundaries, spatial relationships, or when you need to isolate specific objects
   - **Blur Oval/Box**: Best when you have coordinates from previous steps and want precise, focused analysis
   - **OCR Focus**: Best for text-heavy regions like charts, titles, legends

3. **Coordinate Intelligence:**
   - If coordinates are available from previous steps, prefer blur_oval or blur_box for precise focus. 
   - However, if the verifier suggests focus is wrong, avoid using coordinates.
   - Use coordinates to create precise, focused masks that expand on previous findings
   - Coordinate priority: custom_bbox > objects > attention_regions > segmentation

4. **Adaptive Approach:**
   - If previous attention was too focused (small regions), consider using coordinates to create expanded masks or segmentation to find more precise coordinates
   - If segmentation found objects but analysis is too narrow, consider using object coordinates for precise masking
   - If current technique isn't working well, consider switching to a different approach

5. **Query Refinement:**
   - Refine queries to be more specific about what to focus on
   - When using coordinates, mention the coordinate source in the refined query
   - Adapt the query based on what previous steps revealed

6. **Parameter Optimization:**
   - For blur_oval: expand regions slightly for better context (1.1-1.3x)
   - For blur_box: use coordinates directly for precise rectangular focus
   - Adjust blur strength based on mask size and coordinate source

7. **AGGRESSIVE Loop Prevention:**
   - If attention has been used 2+ times, AVOID attention and use coordinate-based approaches
   - If any technique has been used 3+ times, FORCE a different technique
   - Vary your approach aggressively when techniques aren't working

Choose the technique that will provide the most relevant context for the current sub-question. Consider all available information including verifier feedback.\n\nRespond ONLY with a valid JSON object (no prose, no code fences). JSON schema: {\"technique\":\"attention|segmentation|blur_oval|blur_box|ocr_focus\",\"refinedQuery\":\"string\",\"params\":{}}` },
            { role: "user", content: `Question: ${userQuery}\nSub-questions: ${subQuestions.join(" | ")}\nCurrent findings: ${findings.join(" | ") || "none"}\nTechnique context: ${Object.entries(techniqueContext).map(([tech, ctxs]) => `${tech}: ${ctxs.length} findings`).join(" | ") || "none"}\nCoordinate history: ${JSON.stringify(coordinateHistory)}\nCurrent step: ${i + 1}/${iters}\nLast verifier suggestion: ${lastSuggestion || "none"}\n\nPlan the next analysis step. Consider what technique would work best given the available coordinates and previous findings.` }
          ]
          let plan: any = { technique: "attention", refinedQuery: userQuery, params: {} }
          try {
            const plannerResp: any = await callOpenAI(planPrompt, 280)
            let parsedPlan: any = null
            if (plannerResp && typeof plannerResp === "object") {
              if (Array.isArray(plannerResp.choices)) {
                const content = (plannerResp.choices?.[0]?.message?.content || "{}").replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
                try { parsedPlan = JSON.parse(content) } catch {}
              } else if (plannerResp.technique || plannerResp.refinedQuery || plannerResp.params) {
                parsedPlan = plannerResp
              }
            }
            if (!parsedPlan && typeof plannerResp === "string") {
              try { parsedPlan = JSON.parse(plannerResp) } catch {}
            }
            if (parsedPlan) {
              plan = {
                technique: parsedPlan.technique || plan.technique,
                refinedQuery: parsedPlan.refinedQuery || plan.refinedQuery,
                params: parsedPlan.params || {}
              }
            }
          } catch {}
          
          // No hardcoded overrides - let the planner make intelligent decisions
          // Only provide helpful guidance through verifier suggestions
          
          const rq = (plan.refinedQuery || "").toLowerCase()
          plan.params = plan.params || {}
          
          // Enhanced coordinate processing: use coordinates from previous steps intelligently
          if (plan.technique === "blur_box" || plan.technique === "blur_oval") {
            // Extract coordinates from coordinate history
            let bestCoords: number[] | null = null
            let coordSource = ""
            
            // Priority: custom_bbox > objects > attention_regions > segmentation
            if (coordinateHistory.custom_bbox && coordinateHistory.custom_bbox.length > 0) {
              bestCoords = coordinateHistory.custom_bbox[coordinateHistory.custom_bbox.length - 1].bbox
              coordSource = "custom_bbox"
            } else if (coordinateHistory.objects && coordinateHistory.objects.length > 0) {
              bestCoords = coordinateHistory.objects[coordinateHistory.objects.length - 1].bbox
              coordSource = "objects"
            } else if (coordinateHistory.attention && coordinateHistory.attention.length > 0) {
              // Use the highest-saliency attention region (array is sorted by server)
              const bestAttention = coordinateHistory.attention[0]
              if (bestAttention.bbox && bestAttention.bbox.length === 4) {
                bestCoords = bestAttention.bbox
                coordSource = "attention"
              }
            } else if (coordinateHistory.segmentation && coordinateHistory.segmentation.length > 0) {
              bestCoords = coordinateHistory.segmentation[coordinateHistory.segmentation.length - 1].bbox
              coordSource = "segmentation"
            }
            
            // If we have coordinates, use them intelligently
            if (bestCoords && bestCoords.length === 4) {
              const [x1, y1, x2, y2] = bestCoords
              
              // For oval masking, create precise regions based on coordinate source
              if (plan.technique === "blur_oval") {
                // Create precise oval regions that capture the full object/region
                let expandFactor = 1.1
                
                if (coordSource === "attention") {
                  // For attention coordinates, create a more focused oval that captures the high-attention area
                  // Don't expand too much to maintain focus on the specific region
                  expandFactor = 1.05
                } else if (coordSource === "objects") {
                  // For object coordinates, expand slightly to include context around the object
                  expandFactor = 1.15
                } else if (coordSource === "segmentation") {
                  // For segmentation coordinates, use the precise boundaries
                  expandFactor = 1.0
                }
                
                const centerX = (x1 + x2) / 2
                const centerY = (y1 + y2) / 2
                const width = (x2 - x1) * expandFactor
                const height = (y2 - y1) * expandFactor

                // Use pixel coordinates (server clamps to image bounds)
                plan.params.x1 = Math.round(Math.max(0, centerX - width / 2))
                plan.params.y1 = Math.round(Math.max(0, centerY - height / 2))
                plan.params.x2 = Math.round(centerX + width / 2)
                plan.params.y2 = Math.round(centerY + height / 2)
                
                // Keep blur strength as provided or default later
              } else {
                // For box masking, use pixel coordinates directly (server will clamp)
                plan.params.x1 = Math.round(Math.max(0, x1))
                plan.params.y1 = Math.round(Math.max(0, y1))
                plan.params.x2 = Math.round(x2)
                plan.params.y2 = Math.round(y2)
                plan.params.blur_strength = 15
              }
              
              // Add coordinate source to refined query for context
              if (!plan.refinedQuery.includes(coordSource)) {
                plan.refinedQuery = `${plan.refinedQuery} (using ${coordSource} coordinates)`
              }
            }
            
            // Special case: if we have multiple coordinate sources and need better coverage
            if (!bestCoords && coordinateHistory.attention && coordinateHistory.attention.length > 0 && 
                coordinateHistory.segmentation && coordinateHistory.segmentation.length > 0) {
              // Combine attention and segmentation coordinates for wider coverage
              const attentionCoords = coordinateHistory.attention[coordinateHistory.attention.length - 1].bbox
              const segCoords = coordinateHistory.segmentation[coordinateHistory.segmentation.length - 1].bbox
              
              if (attentionCoords && attentionCoords.length === 4 && segCoords && segCoords.length === 4) {
                // Create a bounding box that encompasses both regions
                const combinedX1 = Math.min(attentionCoords[0], segCoords[0])
                const combinedY1 = Math.min(attentionCoords[1], segCoords[1])
                const combinedX2 = Math.max(attentionCoords[2], segCoords[2])
                const combinedY2 = Math.max(attentionCoords[3], segCoords[3])
                
                // Expand slightly for better context
                const centerX = (combinedX1 + combinedX2) / 2
                const centerY = (combinedY1 + combinedY2) / 2
                const width = (combinedX2 - combinedX1) * 1.1
                const height = (combinedY2 - combinedY1) * 1.1

                // Use pixel coordinates; clamp happens on server
                plan.params.x1 = Math.round(Math.max(0, centerX - width / 2))
                plan.params.y1 = Math.round(Math.max(0, centerY - height / 2))
                plan.params.x2 = Math.round(centerX + width / 2)
                plan.params.y2 = Math.round(centerY + height / 2)
                plan.params.blur_strength = 15
                
                plan.refinedQuery = `${plan.refinedQuery} (using combined attention+segmentation coordinates)`
                coordSource = "combined"
              }
            }
          }
          
          // Do not override LLM's technique choice with heuristics; rely on the plan as given
          
          // Enhanced spatial bias detection
          if (!plan.params.spatial_bias) {
            if (rq.includes("left")) plan.params.spatial_bias = "left"
            else if (rq.includes("right")) plan.params.spatial_bias = "right"
            else if (rq.includes("top") || rq.includes("upper")) plan.params.spatial_bias = "top"
            else if (rq.includes("bottom") || rq.includes("lower")) plan.params.spatial_bias = "bottom"
            else if (rq.includes("center") || rq.includes("middle")) plan.params.spatial_bias = "center"
          }
          
          // Enhanced bbox bias based on spatial bias
          if (!plan.params.bbox_bias && plan.params.spatial_bias) {
            switch (plan.params.spatial_bias) {
              case "left": plan.params.bbox_bias = [0.0, 0.0, 0.5, 1.0]; break
              case "right": plan.params.bbox_bias = [0.5, 0.0, 1.0, 1.0]; break
              case "top": plan.params.bbox_bias = [0.0, 0.0, 1.0, 0.5]; break
              case "bottom": plan.params.bbox_bias = [0.0, 0.5, 1.0, 1.0]; break
              case "center": plan.params.bbox_bias = [0.25, 0.25, 0.75, 0.75]; break
            }
            plan.params.bias_strength = plan.params.bias_strength ?? 1.3
          }
          send({ type: "plan", step: i + 1, technique: plan.technique, refinedQuery: plan.refinedQuery, params: plan.params, narrative: `Running ${plan.technique} to look for ${plan.refinedQuery}.` })

          // Optional OCR focus on first step
          if (ocr?.layout?.candidates && i === 0) {
            try {
              const q = userQuery.toLowerCase()
              let region: "title" | "x_axis" | "y_axis" | "legend" | null = null
              if (q.includes("title")) region = "title"
              else if (q.includes("x-axis") || q.includes("x axis") || q.includes("horizontal")) region = "x_axis"
              else if (q.includes("y-axis") || q.includes("y axis") || q.includes("vertical")) region = "y_axis"
              else if (q.includes("legend")) region = "legend"
              if (region) {
                const of = await runOCRFocus(imagePath, region)
                lastProcessed = of.processedImageData
                send({ type: "image", step: i + 1, technique: "segmentation", refinedQuery: `${region} text region`, processedImageData: of.processedImageData, narrative: `Focusing around ${region.replace("_","-")}` })
              }
            } catch {}
          }

          // Execute tool
          let processed = ""
          if (plan.technique === "segmentation") {
            // Enhanced segmentation prompt with specific object targets
            const segQuery = objectTargets.length > 0 
              ? `${plan.refinedQuery || userQuery}. Specifically look for: ${objectTargets.join(", ")}`
              : plan.refinedQuery || userQuery
            
            const seg = await runSegmentation(imagePath, segQuery, plan.params || {})
            processed = seg.processedImageData
            
            // Store coordinates and object information
            if (Array.isArray(seg.objects) && seg.objects.length) {
              coordinateHistory.segmentation = seg.objects.map(obj => ({
                bbox: obj.bbox,
                label: obj.label,
                step: i + 1
              }))
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + `objects:${seg.objects.length}`
            }
            // Emit object coords for LLM awareness
            send({ type: "coords", step: i + 1, objects: seg.objects || [] })
          } else if (plan.technique === "blur_box") {
            const { x1, y1, x2, y2, blur_strength = 15 } = plan.params
            if (typeof x1 === 'number' && typeof y1 === 'number' && typeof x2 === 'number' && typeof y2 === 'number') {
              const box = await runBlurBox(imagePath, x1, y1, x2, y2, blur_strength)
              processed = box.processedImageData
              send({ type: "coords", step: i + 1, custom_bbox: box.bbox, technique: "blur_box" })
            } else {
              throw new Error("blur_box requires x1, y1, x2, y2 coordinates in params")
            }
          } else if (plan.technique === "blur_oval") {
            const { x1, y1, x2, y2, blur_strength = 15 } = plan.params
            if (typeof x1 === 'number' && typeof y1 === 'number' && typeof x2 === 'number' && typeof y2 === 'number') {
              const oval = await runBlurOval(imagePath, x1, y1, x2, y2, blur_strength)
              processed = oval.processedImageData
              send({ type: "coords", step: i + 1, custom_bbox: oval.bbox, technique: "blur_oval" })
            } else {
              throw new Error("blur_oval requires x1, y1, x2, y2 coordinates in params")
            }
          } else {
            // Enhanced attention prompt with specific object targets
            const attnQuery = objectTargets.length > 0 
              ? `${plan.refinedQuery || userQuery}. Focus on: ${objectTargets.join(", ")}`
              : plan.refinedQuery || userQuery
            
            const attn = await runAttention(imagePath, attnQuery, plan.params || {})
            processed = attn.processedImageData
            
            // Store attention coordinates (preserve order from server: highest saliency first)
            if (Array.isArray((attn as any).attention_regions)) {
              coordinateHistory.attention = (attn as any).attention_regions.map((r: any) => ({
                bbox: r.bbox,
                polygon: r.polygon,
                mean: r.mean,
                area_fraction: r.area_fraction,
                step: i + 1
              }))
              send({ type: "coords", step: i + 1, attention_regions: (attn as any).attention_regions })
              // Check if attention regions are too small/thin and suggest expansion
              const totalArea = (attn as any).attention_regions.reduce((sum: number, r: any) => sum + (r.area_fraction || 0), 0)
              if (totalArea < 0.15) { // If total attention area is less than 15% of image
                lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "ATTENTION_TOO_NARROW:USE_COORDINATES_TO_EXPAND"
              }
            } else if ((attn as any).attention_bbox) {
              coordinateHistory.attention = [{
                bbox: (attn as any).attention_bbox,
                step: i + 1
              }]
              send({ type: "coords", step: i + 1, attention_bbox: (attn as any).attention_bbox })
            }
          }
          lastProcessed = processed
          lastTechnique = plan.technique
          lastParams = plan.params
          send({ type: "image", step: i + 1, technique: plan.technique, refinedQuery: plan.refinedQuery, processedImageData: processed })

          // Verify
          const sysVerifier = { role: "system", content: `You are a constructive vision verifier. Evaluate if the current view provides sufficient context for the answer. Consider missing context that could be added. 

IMPORTANT EVALUATION CRITERIA:
1. **Technique Effectiveness**: 
   - If attention regions are too small (<15% of image), suggest coordinate-based masking for better focus
   - If segmentation finds objects but focus is too narrow, suggest using coordinates to create expanded oval/box masks
   - If current technique isn't providing enough context, suggest switching approaches

2. **Coordinate Utilization**:
   - If coordinates are available from previous steps, suggest using them for precise masking
   - When attention is too focused, suggest using attention coordinates to create expanded oval masks
   - When segmentation finds objects, suggest using object coordinates for precise masking

3. **Context Coverage**:
   - Evaluate if the current view covers enough of the relevant image area
   - Suggest expanding focus when analysis is too narrow
   - Suggest combining multiple coordinate sources for better coverage

4. **Technique Adaptation**:
   - Suggest switching to coordinate-based approaches when current methods aren't working
   - Suggest using blur_oval for expanded context or blur_box for precise rectangular focus
   - Suggest varying approaches when stuck in ineffective patterns

5. **Balanced Technique Suggestions**:
   - Consider when segmentation might be better than coordinate-based approaches
   - Suggest segmentation for object detection when precise boundaries are needed
   - Suggest coordinate-based approaches when you have good coordinates and need focused analysis
   - Balance between different techniques based on the specific analysis needs

5. **Helpful Guidance**:
   - Provide specific, actionable suggestions for improvement
   - Consider the available coordinates and suggest how to use them effectively
   - Help guide the system toward better analysis without forcing decisions

Return JSON:
{
  "score": 0..1,
  "good": true|false,
  "suggestion": "specific suggestion for improvement",
  "narrative": "one short reason for next action"
}` }
          const verifierUser = { role: "user", content: [
            { type: "text", text: `Question: ${userQuery}\nSub-questions: ${subQuestions.join(" | ")}\nCurrent findings: ${findings.join(" | ") || "none"}\nOCR: ${ocr?.text || "<none>"}\nOCR Layout: ${ocrSummary}\nCurrent technique: ${plan.technique}\nRefined query: ${plan.refinedQuery}\nParams: ${JSON.stringify(plan.params)}${lastSuggestion ? "\nNotes: "+lastSuggestion : ""}\n\nEvaluate what context is missing and suggest diverse approaches for remaining sub-questions.` },
            { type: "image_url", image_url: { url: processed } },
          ]}
          let verdict: any = { score: 0.0, good: false }
          try {
            const v: any = await callOpenAI([sysVerifier, verifierUser], 220)
            if (v && typeof v === "object" && !Array.isArray(v) && ("score" in v || "good" in v)) {
              verdict = v
            } else {
              const content = (v?.choices?.[0]?.message?.content || "{}").replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
              verdict = JSON.parse(content)
            }
          } catch {}
          lastSuggestion = verdict.suggestion || null
          
          // Check if the agent might be focusing in the wrong place
          if (verdict.score < 0.4 && i >= 1) {
            // If score is very low, suggest the agent might be looking in the wrong area
            if (!lastSuggestion?.includes("wrong_place") && !lastSuggestion?.includes("different_focus")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "wrong_focus_area:try_different_region"
            }
          }
          
          // Check for potential blurry image issues that suggest wrong focus
          if (verdict.score < 0.3 && i >= 1) {
            // Very low score might indicate the image is too blurry or the agent is completely off-target
            if (!lastSuggestion?.includes("too_blurry") && !lastSuggestion?.includes("completely_off_target")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "image_too_blurry:agent_wrong_place"
            }
          }
          
          // Extract findings from current step and accumulate them
          try {
            const findingsPrompt = [
              { role: "system", content: "Extract 2-3 key factual findings about what is visible in the image. Focus on spatial relationships, object locations, visible details, colors, shapes, or any other observable information. Be specific about what you can see and where things are located. Do NOT mention analysis methods, blurriness, or focus areas. Return as a single concise sentence." },
              { role: "user", content: `Question: ${plan.refinedQuery}\nTechnique: ${plan.technique}\nVerifier score: ${verdict.score}\nVerifier feedback: ${verdict.suggestion || "none"}\n\nWhat key information is visible in this image? Focus on what you can see and where things are located.` }
            ]
            const findingsResp = await callOpenAI(findingsPrompt, 200)
            const finding = (findingsResp.choices?.[0]?.message?.content || "").trim()
            if (finding && !findings.includes(finding)) {
              findings.push(finding)
              // Track context by technique
              if (!techniqueContext[plan.technique]) {
                techniqueContext[plan.technique] = []
              }
              techniqueContext[plan.technique].push(finding)
            }
          } catch {}
          
          send({ type: "verdict", step: i + 1, score: verdict.score, good: verdict.good, suggestion: verdict.suggestion, narrative: verdict.narrative })
          
          // Check if we have enough findings to answer the main question
          if (verdict.good && verdict.score >= 0.8) break
          if (findings.length >= subQuestions.length && i >= Math.min(3, iters - 1)) break
          
          // Continue gathering context even with low scores, but limit iterations
          if (i >= iters - 1) break
          
          // Always try to extract some context, even from low-scoring steps
          if (verdict.score < 0.4 && findings.length < 2) {
            // Force at least one more attempt to gather context
            if (i < iters - 2) continue
          }
          
          // Incentivize trying oval/box masking if approaches aren't working well
          if (verdict.score < 0.5 && i >= 2) {
            if (!lastSuggestion?.includes("oval") && !lastSuggestion?.includes("box")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "try_oval_box_masking:low_score"
            }
          }
          
          // Suggest custom oval/box masking if we have good coordinates but need more precision
          if (verdict.score >= 0.6 && verdict.score < 0.8 && i >= 1) {
            const hasGoodCoords = (lastProcessed && lastProcessed !== imageData) || 
                                 (plan.technique === "attention" && plan.params.spatial_bias) ||
                                 (plan.technique === "segmentation" && plan.params.mask_type)
            if (hasGoodCoords && !lastSuggestion?.includes("custom_mask")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "try_custom_mask:refine_focus"
            }
          }
          
          // Suggest expanding attention region if score is low and we might be too focused
          if (verdict.score < 0.5 && i >= 1) {
            if (!lastSuggestion?.includes("expand_attention") && !lastSuggestion?.includes("wider_focus")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "expand_attention_region:wider_context"
            }
          }
          
          // Auto-suggest oval masking when we have segmentation coordinates
          if (verdict.score < 0.6 && coordinateHistory.segmentation && coordinateHistory.segmentation.length > 0 && i >= 1) {
            if (!lastSuggestion?.includes("oval") && !lastSuggestion?.includes("blur_oval")) {
              const lastSeg = coordinateHistory.segmentation[coordinateHistory.segmentation.length - 1]
              if (lastSeg.bbox && lastSeg.bbox.length === 4) {
                // Convert normalized coordinates to pixel coordinates for oval masking
                const [x1, y1, x2, y2] = lastSeg.bbox
                lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + `COORDINATES_AVAILABLE:USE_BLUR_OVAL_FOR_PRECISE_FOCUS_${x1.toFixed(2)}_${y1.toFixed(2)}_${x2.toFixed(2)}_${y2.toFixed(2)}`
              }
            }
          }
          
          // Enhanced technique switching with coordinate intelligence
          if (verdict.score < 0.4 && i >= 2) {
            if (!lastSuggestion?.includes("different_technique")) {
              const currentTech = plan.technique
              let alternativeTech = "attention"
              let reasoning = ""
              
              // Intelligent technique selection based on available coordinates and current performance
              if (currentTech === "attention") {
                if (coordinateHistory.attention && coordinateHistory.attention.length > 0) {
                  // If attention found regions but they're too small, use coordinates to create expanded masks
                  alternativeTech = "blur_oval"
                  reasoning = "ATTENTION_TOO_NARROW:USE_COORDINATES_TO_EXPAND"
                } else {
                  // Try segmentation to find objects
                  alternativeTech = "segmentation"
                  reasoning = "TECHNIQUE_FAILING:SWITCH_TO_SEGMENTATION"
                }
              } else if (currentTech === "segmentation") {
                if (coordinateHistory.objects && coordinateHistory.objects.length > 0) {
                  // If segmentation found objects, use their coordinates for precise masking
                  alternativeTech = "blur_oval"
                  reasoning = "COORDINATES_AVAILABLE:USE_BLUR_OVAL_FOR_PRECISE_FOCUS"
                } else {
                  // Try attention with different parameters
                  alternativeTech = "attention"
                  reasoning = "TECHNIQUE_FAILING:SWITCH_TO_ATTENTION"
                }
              } else if (currentTech === "blur_oval") {
                // Try box masking for more precise rectangular focus
                alternativeTech = "blur_box"
                reasoning = "TECHNIQUE_FAILING:SWITCH_TO_BLUR_BOX"
              } else if (currentTech === "blur_box") {
                // If box masking isn't working, try attention with spatial bias
                alternativeTech = "attention"
                reasoning = "box_too_rigid:try_attention_with_spatial_bias"
              }
              
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + `try_${alternativeTech}:${reasoning}`
            }
          }
          
          // Helpful suggestions for improving analysis
          if (verdict.score < 0.5 && plan.technique === "attention" && i >= 1) {
            if (!lastSuggestion?.includes("try_segmentation") && !lastSuggestion?.includes("segmentation")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "try_segmentation:attention_not_working"
            }
          }
          
          // Suggest using coordinates when they're available for better focus
          if (verdict.score < 0.6 && coordinateHistory.segmentation && coordinateHistory.segmentation.length > 0 && i >= 1) {
            if (!lastSuggestion?.includes("use_coordinates") && !lastSuggestion?.includes("precise_mask")) {
              const lastSeg = coordinateHistory.segmentation[coordinateHistory.segmentation.length - 1]
              if (lastSeg.bbox && lastSeg.bbox.length === 4) {
                lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + `use_coordinates:precise_oval_mask_${lastSeg.bbox.join('_')}`
              }
            }
          }
          
          // Suggest using attention coordinates for expanded masking when attention is too focused
          if (verdict.score < 0.5 && coordinateHistory.attention && coordinateHistory.attention.length > 0 && i >= 1) {
            const lastAttention = coordinateHistory.attention[coordinateHistory.attention.length - 1]
            if (lastAttention.area_fraction && lastAttention.area_fraction < 0.15) {
              if (!lastSuggestion?.includes("expand_attention_coords")) {
                lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "expand_attention_coords:attention_too_focused_use_coordinates_for_wider_mask"
              }
            }
          }
          
          // Suggest combining multiple coordinate sources for better coverage
          if (verdict.score < 0.6 && i >= 2) {
            const hasMultipleCoordSources = (coordinateHistory.attention && coordinateHistory.attention.length > 0) &&
                                          (coordinateHistory.segmentation && coordinateHistory.segmentation.length > 0)
            if (hasMultipleCoordSources && !lastSuggestion?.includes("combine_coordinates")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "combine_coordinates:merge_attention_and_segmentation_for_better_coverage"
            }
          }
          
          // Suggest using object coordinates for precise masking when segmentation found objects
          if (verdict.score < 0.6 && coordinateHistory.objects && coordinateHistory.objects.length > 0 && i >= 1) {
            if (!lastSuggestion?.includes("use_object_coords") && !lastSuggestion?.includes("precise_object_mask")) {
              const lastObj = coordinateHistory.objects[coordinateHistory.objects.length - 1]
              if (lastObj.bbox && lastObj.bbox.length === 4) {
                lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + `use_object_coords:precise_mask_around_${lastObj.label || 'object'}`
              }
            }
          }
          
          // Suggest switching to coordinate-based approaches when current methods are too limited
          if (verdict.score < 0.5 && i >= 2 && (plan.technique === "attention" || plan.technique === "segmentation")) {
            const hasCoords = (coordinateHistory.attention && coordinateHistory.attention.length > 0) ||
                             (coordinateHistory.objects && coordinateHistory.objects.length > 0) ||
                             (coordinateHistory.segmentation && coordinateHistory.segmentation.length > 0)
            if (hasCoords && !lastSuggestion?.includes("switch_to_coordinates")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "switch_to_coordinates:use_available_coordinates_for_precise_focus"
            }
          }
          
          // Helpful guidance for technique variation
          if (i >= 2 && plan.technique === "attention") {
            const attentionCount = techniqueContext.attention ? techniqueContext.attention.length : 0
            if (attentionCount >= 2 && !lastSuggestion?.includes("consider_different_approach")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "consider_different_approach:attention_used_multiple_times"
            }
          }
          
          // Suggest technique variation when stuck in patterns
          if (i >= 3) {
            const currentTechCount = techniqueContext[plan.technique] ? techniqueContext[plan.technique].length : 0
            if (currentTechCount >= 3 && !lastSuggestion?.includes("vary_technique")) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "vary_technique:consider_different_approach_for_better_results"
            }
          }
          
          // Suggest using available coordinates more effectively
          if (i >= 2) {
            const hasUnusedCoords = (coordinateHistory.attention && coordinateHistory.attention.length > 0) ||
                                   (coordinateHistory.objects && coordinateHistory.objects.length > 0) ||
                                   (coordinateHistory.segmentation && coordinateHistory.segmentation.length > 0)
            
            if (hasUnusedCoords && plan.technique !== "blur_oval" && plan.technique !== "blur_box") {
              if (!lastSuggestion?.includes("consider_coordinates")) {
                lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + "consider_coordinates:coordinates_available_for_better_focus"
              }
            }
          }
        }

        // Final answer on lastProcessed
        const finalImage = lastProcessed || imageData
        const finalSys = { role: "system", content: "You are a precise image analyst. Provide a direct, confident answer to the user's question. NEVER mention: blurred areas, blurriness, focus areas, segmentation, attention, analysis techniques, or any internal processing methods. NEVER say you are 'unable to determine' or that something is 'difficult to analyze'. Instead, synthesize all available information into a clear, definitive response. Use the coordinate data and object information to provide spatially accurate answers. If certain details are unclear, focus on what IS visible and make reasonable inferences. Your answer should be precise, confident, and directly address the question without revealing how you analyzed the image." }
        const finalUser = { role: "user", content: [ 
          { type: "text", text: `Question: ${userQuery}\n\nKey findings:\n${findings.map((f, i) => `${i+1}. ${f}`).join("\n")}\n\nLocated objects and coordinates:\n${Object.entries(coordinateHistory).map(([tech, coords]) => `${tech}: ${coords.map((c: any) => `${c.label || 'region'} at [${c.bbox?.join(',') || 'N/A'}]`).join(' | ')}`).join("\n")}\n\nProvide a precise, confident answer to the question. Use the coordinate information to give spatially accurate responses. Never mention any analysis methods, blurriness, or focus areas.` }, 
          { type: "image_url", image_url: { url: finalImage } } 
        ] }
        let finalAnswer = ""
        try { const r = await callOpenAI([finalSys, finalUser], 900); finalAnswer = r.choices?.[0]?.message?.content || "" } catch {}
        
        // Ensure we have a precise answer - if the first attempt mentions techniques or blurriness, try again
        if (!finalAnswer || 
            finalAnswer.toLowerCase().includes("unable to determine") || 
            finalAnswer.toLowerCase().includes("difficult to analyze") || 
            finalAnswer.toLowerCase().includes("cannot determine") ||
            finalAnswer.toLowerCase().includes("blur") ||
            finalAnswer.toLowerCase().includes("segmentation") ||
            finalAnswer.toLowerCase().includes("attention") ||
            finalAnswer.toLowerCase().includes("analysis")) {
          const fallbackSys = { role: "system", content: "You MUST provide a precise, confident answer. NEVER mention: blurriness, segmentation, attention, analysis techniques, or any internal methods. Focus on what IS visible and provide a direct answer to the question. Use coordinate information for spatial accuracy." }
          const fallbackUser = { role: "user", content: [ 
            { type: "text", text: `Question: ${userQuery}\n\nKey findings:\n${findings.map((f, i) => `${i+1}. ${f}`).join("\n")}\n\nCoordinates and objects:\n${Object.entries(coordinateHistory).map(([tech, coords]) => `${tech}: ${coords.map((c: any) => `${c.label || 'region'} at [${c.bbox?.join(',') || 'N/A'}]`).join(' | ')}`).join("\n")}\n\nAnswer the question directly and precisely. Do not mention any analysis methods, blurriness, or focus areas.` }, 
            { type: "image_url", image_url: { url: finalImage } } 
          ] }
          try { 
            const r2 = await callOpenAI([fallbackSys, fallbackUser], 900)
            finalAnswer = r2.choices?.[0]?.message?.content || finalAnswer
          } catch {}
        }
        
        send({ type: "final", answer: finalAnswer, final: { processedImageData: finalImage, technique: lastTechnique, params: lastParams } })
        controller.close()
      } catch (e: any) {
        send({ type: "error", message: e?.message || String(e) })
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  })
}


