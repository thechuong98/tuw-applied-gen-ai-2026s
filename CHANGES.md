# Changes

## Feedback received at the poster session

**Feedback 1:** The poster could be visually improved, but the interface and workings look great.

**Feedback 2:** The reviewer would have liked to see the authors' perspective on the security of the system's infrastructure itself. Overall: the project is sound, agent roles are distinct, the idea is well thought-out and relevant.

---

## What we changed and why

**Poster redesign**
We improved the visual layout: sections are divided more  obvious now, the agent flow uses consistent colours across the diagram and table, utility scores are displayed as bar charts, and the Learnings and Limitations section is reformatted as bullet points.

**Local Ollama support added**
We added full support for running the system locally with Ollama.

**Infrastructure security note**
In response to feedback 2, we added a note in the README and on the poster acknowledging that the system is a research prototype: no API authentication, input goes to a third-party LLM provider. We highlight Ollama as the recommended option when data privacy is a concern.
