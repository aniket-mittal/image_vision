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
        let lastProcessed: string | null = null
        let lastTechnique: "attention" | "segmentation" = "attention"
        let lastParams: any = {}
        let lastSuggestion: string | null = null

        if (seedImageData) {
          lastProcessed = seedImageData
          send({ type: "image", step: 0, technique: lastTechnique, refinedQuery: userQuery, processedImageData: seedImageData, narrative: "Using user-provided region as initial focus." })
        }

        for (let i = seedImageData ? 1 : 0; i < iters; i++) {
          // Plan
          const sysPlanner = { role: "system", content: "You are a vision planning agent. Return strict JSON with {\"technique\":\"attention|segmentation\",\"refinedQuery\":\"...\",\"params\":{...}}." }
          const plannerUser = { role: "user", content: `Question: ${userQuery}\nOCR: ${ocr?.text || "<none>"}\nOCR Layout: ${ocrSummary}\nPrevious technique: ${lastTechnique || "<none>"}\nPrevious params: ${JSON.stringify(lastParams)}\nVerifier suggestion: ${lastSuggestion || "<none>"}\nUser ROI: ${selection ? JSON.stringify(selection) : "<none>"}` }
          let plan: any = { technique: "attention", refinedQuery: userQuery, params: {} }
          try {
            const plannerResp = await callOpenAI([sysPlanner, plannerUser], 280)
            const content = (plannerResp.choices?.[0]?.message?.content || "{}").replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
            plan = JSON.parse(content)
          } catch {}
          const rq = (plan.refinedQuery || "").toLowerCase()
          plan.params = plan.params || {}
          if (!plan.params.spatial_bias) {
            if (rq.includes("left")) plan.params.spatial_bias = "left"
            else if (rq.includes("right")) plan.params.spatial_bias = "right"
            else if (rq.includes("top") || rq.includes("upper")) plan.params.spatial_bias = "top"
            else if (rq.includes("bottom") || rq.includes("lower")) plan.params.spatial_bias = "bottom"
          }
          if (!plan.params.bbox_bias && plan.params.spatial_bias) {
            switch (plan.params.spatial_bias) {
              case "left": plan.params.bbox_bias = [0.0, 0.0, 0.5, 1.0]; break
              case "right": plan.params.bbox_bias = [0.5, 0.0, 1.0, 1.0]; break
              case "top": plan.params.bbox_bias = [0.0, 0.0, 1.0, 0.5]; break
              case "bottom": plan.params.bbox_bias = [0.0, 0.5, 1.0, 1.0]; break
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
            const seg = await runSegmentation(imagePath, plan.refinedQuery || userQuery, plan.params || {})
            processed = seg.processedImageData
            if (Array.isArray(seg.objects) && seg.objects.length) {
              lastSuggestion = (lastSuggestion ? lastSuggestion + " | " : "") + `objects:${seg.objects.length}`
            }
          } else {
            const attn = await runAttention(imagePath, plan.refinedQuery || userQuery, plan.params || {})
            processed = attn.processedImageData
          }
          lastProcessed = processed
          lastTechnique = plan.technique
          lastParams = plan.params
          send({ type: "image", step: i + 1, technique: plan.technique, refinedQuery: plan.refinedQuery, processedImageData: processed })

          // Verify
          const sysVerifier = { role: "system", content: "You are a vision verifier. Return JSON {\"score\":0..1,\"good\":true|false,\"suggestion\":\"...\"}." }
          const verifierUser = { role: "user", content: [
            { type: "text", text: `Question: ${userQuery}\nOCR: ${ocr?.text || "<none>"}\nOCR Layout: ${ocrSummary}\nCurrent technique: ${plan.technique}\nRefined query: ${plan.refinedQuery}\nParams: ${JSON.stringify(plan.params)}${lastSuggestion ? "\nNotes: "+lastSuggestion : ""}` },
            { type: "image_url", image_url: { url: processed } },
          ]}
          let verdict: any = { score: 0.0, good: false }
          try {
            const v = await callOpenAI([sysVerifier, verifierUser], 220)
            const content = (v.choices?.[0]?.message?.content || "{}").replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
            verdict = JSON.parse(content)
          } catch {}
          lastSuggestion = verdict.suggestion || null
          send({ type: "verdict", step: i + 1, score: verdict.score, good: verdict.good, suggestion: verdict.suggestion })
          if (verdict.good && verdict.score >= 0.8) break
        }

        // Final answer on lastProcessed
        const finalImage = lastProcessed || imageData
        const finalSys = { role: "system", content: "Provide a precise, well-grounded answer as a single paragraph. Do not reveal internal steps." }
        const finalUser = { role: "user", content: [ { type: "text", text: `Question: ${userQuery}` }, { type: "image_url", image_url: { url: finalImage } } ] }
        let finalAnswer = ""
        try { const r = await callOpenAI([finalSys, finalUser], 900); finalAnswer = r.choices?.[0]?.message?.content || "" } catch {}
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


