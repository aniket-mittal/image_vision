import { NextRequest, NextResponse } from "next/server"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || ""

export async function POST(req: NextRequest) {
  try {
    const { instruction, steps } = await req.json()
    if (!instruction) return NextResponse.json({ error: "Missing instruction" }, { status: 400 })
    const prompt = `Write one crisp sentence summarizing what the edit agent just accomplished. Avoid meta words like 'successfully'/'I have'. Be direct and past-tense, e.g. 'Removed the birds and filled the background naturally'.\nInstruction: ${instruction}\nSteps: ${JSON.stringify(steps || [])}`
    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Authorization": `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "Return a single short sentence. No preface, no quotes." },
          { role: "user", content: prompt },
        ],
        max_tokens: 50,
      })
    })
    const j = await r.json()
    const content = j.choices?.[0]?.message?.content?.trim() || "Edit completed."
    return NextResponse.json({ summary: content })
  } catch (e: any) {
    return NextResponse.json({ summary: "Edit completed." })
  }
}


