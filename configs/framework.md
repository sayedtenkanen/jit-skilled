# SKILL.md — Required Structure

This skeleton is FIXED. Every synthesized skill must be a short Markdown
document with exactly these four sections, in this order. Only the content
inside each section changes per task.

## Task Type
One line naming the kind of question this is (e.g. "financial metric lookup",
"policy lookup").

## Retrieval Notes
1-3 bullet points on what the retrieved examples show about how similar
questions were correctly answered. Ground every bullet in the retrieved
examples provided below. Do not invent facts that are not present in them.

## Answering Strategy
2-4 concrete, imperative instructions for how to locate and extract the
answer from the source document for THIS specific question.

## Output Format
One line specifying exactly how the final answer should be formatted (units,
symbols, precision), based on how the retrieved examples' answers were
formatted.

Keep the entire document under 200 words. Do not restate the full source
document or retrieved examples verbatim — synthesize guidance, don't copy.
