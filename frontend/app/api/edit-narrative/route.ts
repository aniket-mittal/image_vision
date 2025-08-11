import { NextRequest, NextResponse } from "next/server"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || ""

export async function POST(req: NextRequest) {
  try {
    const { instruction } = await req.json()
    if (!instruction) return NextResponse.json({ error: "Missing instruction" }, { status: 400 })

    // Ask LLM to draft a concise in-progress narrative for edit tasks
    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Authorization": `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You draft a short, stepwise, present-tense narrative of image editing actions. Output strict JSON only: {\"lines\":[\"...\"]}. Keep each line crisp (<= 10 words)." },
          { role: "user", content: `Instruction: ${instruction}\nReturn JSON with 5-7 lines that reflect a realistic pipeline such as: analyzing intent, building target mask, refining mask, inpainting/generation, smoothing/blending, and validating the result.` },
        ],
        max_tokens: 180,
      })
    })
    const j = await r.json()
    let lines: string[] = []
    try {
      const content = j.choices?.[0]?.message?.content || "{}"
      const clean = String(content).replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
      const parsed = JSON.parse(clean)
      if (Array.isArray(parsed?.lines)) lines = parsed.lines
    } catch {}
    if (!lines.length) {
      lines = [
        "Analyzing edit request",
        "Locating target region",
        "Running segmentation",
        "Refining the mask",
        "Running image generation",
        "Smoothing and blending",
        "Validating result",
      ]
    }
    return NextResponse.json({ lines })
  } catch (e: any) {
    return NextResponse.json({ lines: [
      "Analyzing edit request",
      "Running segmentation",
      "Running image generation",
      "Validating result",
    ] })
  }
}


