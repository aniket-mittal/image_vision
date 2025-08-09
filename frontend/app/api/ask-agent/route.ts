import { NextRequest, NextResponse } from "next/server"
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

interface StepResult {
  processedImageData: string
  technique: "attention" | "segmentation"
  refinedQuery: string
  params: any
  rationale?: string
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
    // Optional spatial hinting for disambiguation like "left bird" vs "right bird"
    spatial_bias: p.spatial_bias,
    bbox_bias: p.bbox_bias,
    output_dir: "temp_output",
  }
  const r = await fetch(`${MODEL_SERVER_URL}/attention`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`attention ${r.status}: ${await r.text()}`)
  const data = await r.json()
  return data.processedImageData as string
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
  const data = await r.json()
  if (!data.processedImageData) throw new Error("no processedImageData from segmentation")
  return data.processedImageData as string
}

async function runOCR(imagePath: string) {
  const body = {
    image_data: await import("fs/promises").then((fs) =>
      fs.readFile(imagePath).then((b) => `data:image/jpeg;base64,${b.toString("base64")}`),
    ),
  }
  const r = await fetch(`${MODEL_SERVER_URL}/ocr`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`ocr ${r.status}: ${await r.text()}`)
  const data = await r.json()
  return data as { text: string; blocks: Array<{ text: string; bbox: number[]; confidence: number }> }
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

function maxIterations(mode: DepthMode) {
  return mode === "BlurDeep" ? 20 : 5
}

export async function POST(req: NextRequest) {
  try {
    const { imageData, userQuery, mode, aiModel, selection, seedImageData } = (await req.json()) as AgentInput
    if (!imageData || !userQuery || !mode) {
      return NextResponse.json({ error: "Missing imageData, userQuery, or mode" }, { status: 400 })
    }

    // Persist input image to tmp
    const tempDir = join(tmpdir(), "image_vision_agent")
    await mkdir(tempDir, { recursive: true })
    const imageBuffer = Buffer.from(imageData.split(",")[1], "base64")
    const imagePath = join(tempDir, "agent_input.jpg")
    await writeFile(imagePath, imageBuffer)

    // OCR context (text to ground questions like street signs)
    let ocr = { text: "", blocks: [] as any[] }
    try {
      ocr = await runOCR(imagePath)
    } catch (e) {
      // non-fatal
      console.warn("OCR failed", e)
    }

    // Prepare OCR summary for prompts
    const ocrSummary = (() => {
      try {
        const c = (ocr as any)?.layout?.candidates || {}
        const title = c?.title ? "yes" : "no"
        const xt = Array.isArray(c?.x_axis_ticks) ? c.x_axis_ticks.length : 0
        const yt = Array.isArray(c?.y_axis_ticks) ? c.y_axis_ticks.length : 0
        const lg = Array.isArray(c?.legend_candidates) ? c.legend_candidates.length : 0
        return `layout -> title:${title}, x_ticks:${xt}, y_ticks:${yt}, legend_boxes:${lg}`
      } catch {
        return "layout -> unknown"
      }
    })()

    // Initialize planner prompt
    const sysPlanner = {
      role: "system",
      content:
        "You are a vision planning agent. You decide, step by step, where to look using attention or segmentation masking. You may tune parameters (blur_strength, padding, mask_type; layer_index, enhancement_control, smoothing_kernel, overlay_strength, grayscale_level). Use concise JSON outputs only. When OCR indicates chart-like structure (title/x-axis/y-axis/legend), exploit spatial relations (e.g., look above/below/left/right of labeled text) to focus the view.",
    }
    const sysVerifier = {
      role: "system",
      content:
        "You are a vision verifier. Given the question and an intermediate masked view, judge if the current view likely answers the question. If not, suggest adjustments. Return strict JSON only.",
    }

    const iters = maxIterations(mode)
    const steps: StepResult[] = []
    let bestAnswer: string | null = null
    let bestRationale: string | null = null
    let bestScore = -1
    let lastProcessed: string | null = null
    let lastRefined = ""
    let lastTechnique: "attention" | "segmentation" = "attention"
    let lastParams: any = {}
    let lastSuggestion: string | null = null

    // If user provided a masked seed image, allow verifier to judge it first
    if (seedImageData) {
      lastProcessed = seedImageData
      // Optionally set a neutral plan
      lastTechnique = "attention"
      lastParams = {}
      steps.push({ processedImageData: seedImageData, technique: lastTechnique, refinedQuery: userQuery, params: {} as any, rationale: "user-seed" })
    }

    for (let i = seedImageData ? 1 : 0; i < iters; i++) {
      // Ask planner what to do next
      const plannerUser = {
        role: "user",
        content: `Question: ${userQuery}\nOCR: ${ocr.text || "<none>"}\nOCR Layout: ${ocrSummary}\nPrevious technique: ${lastTechnique || "<none>"}\nPrevious params: ${JSON.stringify(lastParams)}\nVerifier suggestion: ${lastSuggestion || "<none>"}\nUser provided ROI: ${selection ? JSON.stringify(selection) : "<none>"}\nSub-questions (if any) will be provided later to the final answer. Step ${i + 1} of ${iters}. Respond as JSON: {"technique":"attention|segmentation","refinedQuery":"...","params":{...},"rationale":"..."}`,
      }
      const plannerResp = await callOpenAI([sysPlanner, plannerUser], 400)
      const plannerContent: string = plannerResp.choices?.[0]?.message?.content || "{}"
      const clean = plannerContent.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
      let plan: { technique: "attention" | "segmentation"; refinedQuery: string; params: any; rationale?: string }
      try {
        plan = JSON.parse(clean)
      } catch {
        plan = { technique: "attention", refinedQuery: userQuery, params: {}, rationale: "fallback" }
      }

      // Heuristic: derive spatial_bias when the refined query references sides
      const rq = (plan.refinedQuery || "").toLowerCase()
      if (!plan.params) plan.params = {}
      if (!plan.params.spatial_bias) {
        if (rq.includes("left")) plan.params.spatial_bias = "left"
        else if (rq.includes("right")) plan.params.spatial_bias = "right"
        else if (rq.includes("top") || rq.includes("upper")) plan.params.spatial_bias = "top"
        else if (rq.includes("bottom") || rq.includes("lower")) plan.params.spatial_bias = "bottom"
      }
      if (!plan.params.bbox_bias && plan.params.spatial_bias) {
        // Provide a coarse bbox bias to amplify side-specific focus
        switch (plan.params.spatial_bias) {
          case "left":
            plan.params.bbox_bias = [0.0, 0.0, 0.5, 1.0]
            break
          case "right":
            plan.params.bbox_bias = [0.5, 0.0, 1.0, 1.0]
            break
          case "top":
            plan.params.bbox_bias = [0.0, 0.0, 1.0, 0.5]
            break
          case "bottom":
            plan.params.bbox_bias = [0.0, 0.5, 1.0, 1.0]
            break
        }
      }

      // Special handling for charts/graphs: if OCR suggests structured labels, allow a text-region focus before tools
      if (ocr?.layout?.candidates && i === 0) {
        try {
          // Pick a heuristic region based on question keywords
          const q = userQuery.toLowerCase()
          let region: "title" | "x_axis" | "y_axis" | "legend" | null = null
          if (q.includes("title")) region = "title"
          else if (q.includes("x-axis") || q.includes("x axis") || q.includes("horizontal")) region = "x_axis"
          else if (q.includes("y-axis") || q.includes("y axis") || q.includes("vertical")) region = "y_axis"
          else if (q.includes("legend")) region = "legend"
          if (region) {
            const of = await runOCRFocus(imagePath, region)
            steps.push({ processedImageData: of.processedImageData, technique: "segmentation", refinedQuery: `${region} text region`, params: { region } })
            lastProcessed = of.processedImageData
          }
        } catch {}
      }

      // Execute tool call
      let processed: string
      if (plan.technique === "segmentation") {
        processed = await runSegmentation(imagePath, plan.refinedQuery || userQuery, plan.params || {})
      } else {
        processed = await runAttention(imagePath, plan.refinedQuery || userQuery, plan.params || {})
      }

      lastProcessed = processed
      lastRefined = plan.refinedQuery
      lastTechnique = plan.technique
      lastParams = plan.params

      // Verifier scoring and adjustment
      const verifierUser = {
        role: "user",
        content: [
          { type: "text", text: `Question: ${userQuery}\nOCR: ${ocr.text || "<none>"}\nOCR Layout: ${ocrSummary}\nCurrent technique: ${plan.technique}\nRefined query: ${plan.refinedQuery}\nParams: ${JSON.stringify(plan.params)}\nReturn JSON: {"score":0-1,"good":true|false,"suggestion":"..."}` },
          { type: "image_url", image_url: { url: processed } },
        ],
      }
      const verifierResp = await callOpenAI([sysVerifier, verifierUser], 300)
      const verifierContent: string = verifierResp.choices?.[0]?.message?.content || "{}"
      const vclean = verifierContent.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
      let verdict: { score: number; good: boolean; suggestion?: string }
      try {
        verdict = JSON.parse(vclean)
      } catch {
        verdict = { score: 0.0, good: false, suggestion: "" }
      }
      lastSuggestion = verdict.suggestion || null

      steps.push({ processedImageData: processed, technique: plan.technique, refinedQuery: plan.refinedQuery, params: plan.params, rationale: plan.rationale })

      if (verdict.score > bestScore) {
        bestScore = verdict.score
        bestAnswer = null
        bestRationale = plan.rationale || null
      }

      // Early stop if good enough
      if (verdict.good && verdict.score >= 0.8) break

      // Planner will adapt next iteration implicitly as it receives previous step context
    }

    // Query decomposition: expand the user's question into sub-questions to gather more context
    const decomposeSys = { role: "system", content: "Decompose the user's visual question into the smallest useful sub-questions to fully answer it. Return strict JSON: {\"subQuestions\":[\"...\"]}." }
    const decomposeUser = { role: "user", content: `Question: ${userQuery}` }
    let subQuestions: string[] = []
    try {
      const d = await callOpenAI([decomposeSys, decomposeUser], 250)
      const dc = (d.choices?.[0]?.message?.content || "{}").replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
      const parsed = JSON.parse(dc)
      if (Array.isArray(parsed?.subQuestions)) subQuestions = parsed.subQuestions
    } catch {}

    // Per-subquestion quick investigation (1 pass each) to build hidden context, bounded by mode
    const maxSubQ = Math.min(subQuestions.length, (mode === "BlurDeep" ? 5 : 2))
    const findings: string[] = []
    for (let i = 0; i < maxSubQ; i++) {
      const sq = (subQuestions[i] || "").trim()
      if (!sq) continue
      // Ask planner once for this sub-question
      const sqPlannerUser = { role: "user", content: `Sub-question: ${sq}\nMain question: ${userQuery}\nOCR: ${ocr.text || "<none>"}\nOCR Layout: ${ocrSummary}\nRespond JSON: {\"technique\":\"attention|segmentation\",\"refinedQuery\":\"...\",\"params\":{...}}` }
      let sqPlan: any = { technique: "attention", refinedQuery: sq, params: {} }
      try {
        const sqResp = await callOpenAI([sysPlanner, sqPlannerUser], 250)
        const sqc = (sqResp.choices?.[0]?.message?.content || "{}").replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
        sqPlan = JSON.parse(sqc)
      } catch {}
      const rq2 = (sqPlan.refinedQuery || sq).toLowerCase()
      if (!sqPlan.params) sqPlan.params = {}
      if (!sqPlan.params.spatial_bias) {
        if (rq2.includes("left")) sqPlan.params.spatial_bias = "left"
        else if (rq2.includes("right")) sqPlan.params.spatial_bias = "right"
        else if (rq2.includes("top") || rq2.includes("upper")) sqPlan.params.spatial_bias = "top"
        else if (rq2.includes("bottom") || rq2.includes("lower")) sqPlan.params.spatial_bias = "bottom"
      }
      if (!sqPlan.params.bbox_bias && sqPlan.params.spatial_bias) {
        switch (sqPlan.params.spatial_bias) {
          case "left": sqPlan.params.bbox_bias = [0.0, 0.0, 0.5, 1.0]; break
          case "right": sqPlan.params.bbox_bias = [0.5, 0.0, 1.0, 1.0]; break
          case "top": sqPlan.params.bbox_bias = [0.0, 0.0, 1.0, 0.5]; break
          case "bottom": sqPlan.params.bbox_bias = [0.0, 0.5, 1.0, 1.0]; break
        }
      }
      let sqImage = ""
      try {
        if (sqPlan.technique === "segmentation") sqImage = await runSegmentation(imagePath, sqPlan.refinedQuery, sqPlan.params)
        else sqImage = await runAttention(imagePath, sqPlan.refinedQuery, sqPlan.params)
        steps.push({ processedImageData: sqImage, technique: sqPlan.technique, refinedQuery: sqPlan.refinedQuery, params: sqPlan.params, rationale: `subq: ${sq}` })
      } catch {}
      // Optional: a brief internal finding
      if (sqImage) {
        try {
          const findSys = { role: "system", content: "Given the sub-question and the processed image, extract one concise factual note (no enumeration, <25 words). Return plain text only." }
          const findUser = { role: "user", content: [{ type: "text", text: `Sub-question: ${sq}` }, { type: "image_url", image_url: { url: sqImage } }] }
          const f = await callOpenAI([findSys, findUser], 120)
          const txt = (f.choices?.[0]?.message?.content || "").trim()
          if (txt) findings.push(txt)
        } catch {}
      }
    }

    // Final answering on the best/last processed view
    const finalImage = lastProcessed || imageData
    const finalMessages = [
      {
        role: "system",
        content: `You are an expert vision assistant. Provide a precise, well-grounded answer.
Rules:
- Do NOT expose internal sub-questions or an outline.
- Avoid numbered lists unless explicitly asked.
- Synthesize into a single, coherent answer in natural prose.
If text in the scene matters, incorporate OCR context implicitly.`,
      },
      {
        role: "user",
        content: [
          { type: "text", text: `Question: ${userQuery}\nContext (internal findings, do not verbatim list them): ${findings.join(" | ")}` },
          { type: "image_url", image_url: { url: finalImage } },
        ],
      },
    ]
    const finalResp = await callOpenAI(finalMessages, 1000)
    const finalAnswer: string = finalResp.choices?.[0]?.message?.content || ""

    return NextResponse.json({
      answer: finalAnswer,
      steps,
      final: { processedImageData: finalImage, refinedQuery: lastRefined, technique: lastTechnique, params: lastParams },
      ocr,
      iterations: steps.length,
      mode,
      aiModel: aiModel || "GPT",
    })
  } catch (e: any) {
    console.error("ask-agent error:", e)
    return NextResponse.json({ error: e?.message || "error" }, { status: 500 })
  }
}


