"""
Generates batch_01.jsonl with 100 labeled tweet scoring examples.
Run: python data/raw/batch_01_raw.py
Output: data/raw/batch_01.jsonl
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dataset.schemas import ScoredExample, RUBRIC_HASH  # noqa: E402

def make(id, content, pillar, hook, tone, x_algo, data_s, pillar_s, cta, nvl, reasoning):
    scores = {
        "hook_strength": hook,
        "tone_compliance": tone,
        "x_algorithm_optimization": x_algo,
        "data_specificity": data_s,
        "pillar_alignment": pillar_s,
        "cta_quality": cta,
        "never_list_violation": nvl,
        "reasoning": reasoning,
    }
    # compute tier via schema
    from dataset.schemas import ScoreSet
    ss = ScoreSet(**scores)
    tier = ss.tier()
    return {
        "id": id,
        "content": content,
        "pillar": pillar,
        "quality_tier": tier,
        "scores": scores,
        "rubric_hash": RUBRIC_HASH,
    }

EXAMPLES = [
    # ── READY (composite ≥ 9.25) ─────────────────────────────────────────────
    make("draft_0001",
         "GPT-4 scores 67% on MATH. Fine-tuned Llama-3-8B scores 71%.\n\nBigger isn't always better. The research that changes how you pick models isn't on the leaderboard.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="Hook is falsifiable with named numbers. Tone clean: no hashtags, em-dashes, exclamations. Debate-bait framing drives replies. Named model + concrete benchmark. Pillar obvious in first line. CTA is implicit debate-bait."),

    make("draft_0002",
         "Anthropic's paper on Constitutional AI has 847 citations in 18 months.\n\nEvery RLHF paper published since references it. Read the original, not the summaries.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="847 citations + 18 months is falsifiable. Clean tone. High debate potential around RLHF. Specific paper named. AI research pillar immediate. CTA 'read the original' is TOFU awareness."),

    make("draft_0003",
         "I ran the same RAG pipeline with 3 different embedding models.\n\ntext-embedding-3-large: 84% recall\nbge-large-en: 81% recall\nall-MiniLM-L6: 67% recall\n\nThe cost difference: 12x. The quality difference: not 12x.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific recall numbers across 3 named models. Cost ratio 12x is falsifiable. Clean professional tone. Strong debate: is quality worth 12x cost? Tooling pillar clear in first line. CTA is implied comparison debate."),

    make("draft_0004",
         "I rejected an offer 40% above my current salary.\n\nThe team had shipped 0 production ML in 2 years. The tech debt was a decade old. Salary is negotiable. Momentum is not.",
         "career", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=8, nvl=False,
         reasoning="Falsifiable: 40% salary, 2 years, 0 production. Clean tone. Career debate-bait: would you take 40% for bad team. Pillar immediate. CTA implicit: provokes disagreement."),

    make("draft_0005",
         "Every senior engineer I know has a list of 3 things they will not work on.\n\nMine: consumer social, adtech, anything with 'AI wrapper' in the pitch deck. Yours?",
         "career", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="Hook: universalizes a specific pattern. Clean tone. Strong reply driver: invites audience to share their list. Specific personal examples. Career pillar in first sentence. CTA is debate-bait question."),

    make("draft_0006",
         "LangChain added 4 new abstractions last week.\n\nI removed 2 of them from our codebase. The parts I removed: 800 lines of framework, 90 lines of direct API calls.\n\nAbstractions are a cost, not a feature.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="Named tool, concrete line counts, recent timeframe. Clean professional tone. High debate: LangChain community is defensive. Tooling pillar clear. CTA is implicit take that invites rebuttal."),

    make("draft_0007",
         "Attention is O(n^2) in sequence length.\n\nA 4K token context costs 16x more compute than a 1K context. Every time someone asks for 'longer context windows' they're asking for a compute budget 4x worse than they think.",
         "ai_research", hook=10, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=8, nvl=False,
         reasoning="Visualizable math: O(n^2) is concrete. 16x vs 4x clarifies a widespread misunderstanding. Clean tone. High debate potential. Research pillar immediate. CTA weak but not harmful."),

    make("draft_0008",
         "We replaced our Claude API calls with a fine-tuned Llama-3-8B.\n\nAPI cost went from $2,100/month to $190/month (GPU rental).\nLatency went from 1.2s to 340ms.\n\nFine-tuning isn't academic. It's ops math.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific $ and ms numbers. Named models. Clean tone. Strong debate: fine-tuning ROI is divisive. Tooling pillar immediate. CTA implicit: 'ops math' framing invites disagreement."),

    make("draft_0009",
         "The best debugging session I had this month took 6 hours.\n\nThe bug: a float32/float16 cast silently capping gradients to 0. The lesson: any ML bug that disappears on smaller batches is a numerical precision bug until proven otherwise.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific timeframe, named technical root cause. Clean tone. Highly shareable debugging heuristic. Tooling pillar clear. CTA: 'until proven otherwise' invites rebuttals from people with different experiences."),

    make("draft_0010",
         "I've interviewed 50+ ML engineers in the last 2 years.\n\nThe single question that predicts performance better than any other: 'Walk me through a time your model worked in dev and broke in prod. What did you do?'\n\nCoding tests don't catch this.",
         "career", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="50+ and 2 years are specific. Named exact interview question. Clean tone. High debate: coding tests vs behavioral is perennial. Career pillar first sentence. CTA: implicit challenge to interviewers."),

    make("draft_0011",
         "Mixtral-8x7B outperforms GPT-3.5-Turbo on 11 of 14 benchmark tasks.\n\nCost per million tokens: Mixtral $0.70, GPT-3.5 $3.00.\n\nThe open-source gap is 4 months wide at most.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="Named models, benchmark count, exact pricing. Clean tone. High debate: open-source vs API is tribal. Research pillar immediate. CTA: '4 months wide' is a falsifiable claim people want to refute."),

    make("draft_0012",
         "I wrote 1,200 lines of Python to build a feature. My junior deleted 900 of them and the tests still passed.\n\nI promoted her 6 months early. The ability to delete code is rarer than the ability to write it.",
         "career", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific line counts + promotion timeframe. Clean tone. Counterintuitive: deletion is a sign of seniority. Career pillar in first sentence. CTA implicit: invites managers to agree or debate."),

    make("draft_0013",
         "Prompt engineering has a half-life of 6 months.\n\nEvery trick that worked with GPT-3.5 broke on GPT-4. Most GPT-4 tricks don't transfer to Claude 3. Build the eval harness, not the prompt library.",
         "takes", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="Half-life metaphor is visualizable. Named models. Clean tone. Strong take: prompt engineering as a dead end. Takes pillar. CTA: 'build eval harness' is actionable and controversial."),

    make("draft_0014",
         "Vector search returns 10 results. Your LLM reads all 10 and answers from the 3 that matter.\n\nThe other 7 are noise that degrades output quality. Most RAG pipelines never measure this. Ours did: removing irrelevant chunks cut hallucination rate by 34%.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific numbers: 10/7/3/34%. Named mechanism (RAG). Clean tone. Actionable: measure chunk relevance. Tooling pillar immediate. CTA: implicit challenge to build the measurement."),

    make("draft_0015",
         "The paper 'Chain of Thought Prompting Elicits Reasoning in LLMs' has been cited 6,000+ times in 2 years.\n\nMost citations are from people who added 'think step by step' and called it research. The original insight: CoT only works above ~100B parameters.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="6000 citations, 2 years, 100B param threshold all specific. Clean tone. Debate: calling out low-effort citations. Research pillar immediate. CTA: implicit 'check your parameter count' challenge."),

    make("draft_0016",
         "Most AI startups will die when GPT-5 ships.\n\nNot because OpenAI wins. Because GPT-5 closes the gap their product existed to fill. If your moat is 'better prompting of GPT-4', you don't have a moat.",
         "takes", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific prediction: GPT-5 as catalyst. Clean tone. Highly controversial: will generate debate from founders. Takes pillar clear. CTA: implicit 'check your moat' challenge."),

    make("draft_0017",
         "I've led 3 ML teams. The one that shipped fastest had the worst model accuracy.\n\nThey had daily deploys, feature flags per user segment, and a rollback button that worked. The team with the best models shipped quarterly.",
         "career", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="3 teams, daily deploys vs quarterly—specific. Clean tone. Counterintuitive take on accuracy vs shipping. Career/leadership pillar. CTA: implicit challenge to ML teams optimizing for accuracy."),

    make("draft_0018",
         "Claude 3 Haiku processes 1 million tokens for $0.25.\n\nFor context: that's 750,000 words. The average novel is 90,000 words. You can process 8 full novels for a quarter.\n\nThe cost argument against AI products is gone.",
         "takes", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=8, nvl=False,
         reasoning="Named model, exact pricing, concrete word/book comparisons. Clean tone. Controversial claim: cost argument is gone. Takes pillar. CTA weak but numbers make the tweet strong."),

    make("draft_0019",
         "Cursor AI increased my coding speed by 40% on boilerplate. By 0% on architecture decisions.\n\nThe productivity gains from AI tools are real. They're just not where the hype says they are.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="Named tool, two specific percentages, specific claim types. Clean tone. Debate: AI productivity measurement. Tooling pillar. CTA: 'not where hype says' invites disagreement."),

    make("draft_0020",
         "I trained a sentiment classifier on 100K labeled examples. It hit 94% accuracy.\n\nA GPT-4 zero-shot prompt hit 91%. The 3-point gap cost me 6 weeks and $12,000 in labeling.\n\nLabeled data is a premium product. Know when to buy it.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="100K examples, 94%/91% accuracy, 6 weeks, $12K all specific. Clean tone. High debate: fine-tuning ROI. Tooling pillar. CTA: 'know when to buy it' is actionable advice."),

    make("draft_0021",
         "The best ML paper I read this quarter was a negative result.\n\nThe team trained 6 model variants, none outperformed the baseline, and they published anyway. That's what scientific integrity looks like. Most teams bury this.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific: 6 variants, negative result. Clean tone. Contrarian: most teams bury negatives. Research pillar. CTA: implicit challenge to publish negative results."),

    make("draft_0022",
         "My team spent 3 months building a custom chunking algorithm for RAG.\n\nWe tested it against fixed 512-token chunks. The custom version won by 2.1 percentage points on recall.\n\nWas it worth 3 months? Probably not. Run the baseline first.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="3 months, 512 tokens, 2.1% gain all specific. Clean tone. Strong self-deprecating honesty. Tooling. CTA: 'run the baseline first' is actionable engineering advice."),

    make("draft_0023",
         "The interview question that filters senior ML engineers from mid-level:\n\n'Your model's accuracy dropped 8% overnight. What are the first 3 things you check?'\n\nPeople who answer 'retrain it' are mid-level. People who answer 'data pipeline' are senior.",
         "career", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific 8% drop, named exact question, tiered answers. Clean tone. High debate: interviewers will argue about the answers. Career pillar. CTA: debate-bait on the correct answer."),

    make("draft_0024",
         "Reinforcement learning from human feedback has one problem nobody talks about:\n\nThe humans doing the feedback are contractors paid $12/hour. Their preference model shapes $100M models. We have no idea how to fix this.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="$12/hour and $100M are specific. Names RLHF clearly. Clean tone. Provocative systemic critique. Research pillar. CTA: 'we have no idea' invites researchers to respond."),

    make("draft_0025",
         "I've used 14 different AI coding assistants over the past 18 months.\n\nThe one thing that separates the top 3 from the rest: they know when to stop generating and ask a clarifying question.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="14 tools, 18 months specific. Clean tone. Specific differentiator. Tooling pillar. CTA: debate-bait on which 3 tools he's referring to."),

    make("draft_0026",
         "Llama 3 70B on a single A100 runs 28 tokens/second.\n\nFor a typical scoring task: 150 token input, 80 token output. That's 3.7 seconds per call. You need 2 A100s to hit under 2 seconds.\n\nSelf-hosting math is always more expensive than the marketing says.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="Named model, specific tokens/sec, exact latency math. Clean tone. Demystifies self-hosting costs. Tooling. CTA: challenges marketing claims with real numbers."),

    make("draft_0027",
         "The replication crisis hit AI research hard in 2023.\n\nStanford researchers retried 50 published AI claims. 60% failed to reproduce under original conditions. 30% failed even with access to the authors' code.\n\nBenchmarks don't mean what we think they mean.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="50 claims, 60%/30% failure rates from Stanford. Clean tone. Alarming finding that invites debate. Research pillar. CTA: implicit challenge to trust published benchmarks."),

    make("draft_0028",
         "I know an engineer who turned down $380K at a FAANG to join a 10-person ML startup.\n\n18 months later the startup is at $4M ARR and he has 2% equity.\n\nRisk calibration is a skill. Most people are bad at it in both directions.",
         "career", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="$380K, $4M ARR, 2% equity, 18 months all specific. Clean tone. High engagement: salary debate. Career pillar. CTA: 'both directions' is balanced and debate-friendly."),

    make("draft_0029",
         "The AI wrapper company playbook: call OpenAI, add a UI, charge 5x margin.\n\nIt worked in 2022. It works less in 2023. It won't work in 2024.\n\nEvery abstraction layer that doesn't own training data is a commodity waiting to be deprecated.",
         "takes", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="Year-specific predictions, named strategy clearly. Clean tone. Provocative: will upset wrapper founders. Takes pillar clear. CTA: 'own training data' is actionable provocation."),

    make("draft_0030",
         "Context windows don't solve the right problem.\n\nThe problem is: given 128K tokens, how do you know which 5K matter?\n\nLong context is a retrieval problem dressed up as a hardware problem.",
         "ai_research", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="Falsifiable reframe: context=retrieval. Specific 128K/5K ratio. Clean tone. Debate: many disagree. Research pillar. CTA: implicit challenge to long-context advocates."),

    make("draft_0031",
         "I deployed 6 ML models to production this year. Only 2 of them are still running.\n\nThe 4 that died: two from data drift, one from API deprecation, one from business pivot.\n\nML deployment is an ongoing cost, not a one-time project.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="6 models, 2 running, named 4 failure modes. Clean tone. Realistic operational view. Tooling pillar. CTA: challenges 'ship and forget' mindset."),

    make("draft_0032",
         "The best career advice I got was from a recruiter who rejected me:\n\n'Your resume shows what you built. It doesn't show what broke in production and what you did about it.'\n\nAdd a postmortem section. Nobody does this.",
         "career", hook=9, tone=9, x_algo=9, data_s=8, pillar_s=9, cta=9, nvl=False,
         reasoning="Specific: rejection, direct quote, novel resume advice. Clean tone. Highly shareable. Career pillar. CTA: actionable and differentiating advice."),

    make("draft_0033",
         "Perplexity beats Google for technical queries 7 out of 10 times in my usage.\n\nNot because it's smarter — because it shows sources, lets me verify, and updates faster.\n\nSearch quality is information trust, not information volume.",
         "takes", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="7/10 specific ratio. Named tools. Clean tone. Debate: Google loyalists will push back. Takes pillar. CTA: reframes quality definition."),

    make("draft_0034",
         "The hardest part of building AI products isn't the model. It's the eval.\n\nWe spent 2 months on a scoring rubric for our AI-generated content. The rubric changed 7 times. Each change invalidated 3 weeks of data.\n\nInvest in evals early. They compound.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=9, pillar_s=9, cta=9, nvl=False,
         reasoning="2 months, 7 rubric changes, 3 weeks data each time—all specific. Clean tone. Actionable: invest in evals early. Tooling pillar. CTA: 'they compound' is a falsifiable claim."),

    make("draft_0035",
         "OpenAI's function calling API changed how I build agents.\n\nBefore: parse LLM output with regex. Failure rate: 12%.\nAfter: structured output schema. Failure rate: 0.3%.\n\nThe reliability jump from unstructured to structured output is not incremental. It's a phase change.",
         "tooling", hook=9, tone=9, x_algo=9, data_s=10, pillar_s=9, cta=9, nvl=False,
         reasoning="12% to 0.3% failure rate is specific. Named API. Clean tone. 'Phase change' is a strong falsifiable claim. Tooling pillar. CTA: implicit challenge to regex parsers."),

    # ── BELOW_TARGET (8.0 ≤ composite < 9.25) ────────────────────────────────
    make("draft_0036",
         "AI tools are changing how developers work.\n\nCursor and GitHub Copilot are the most popular right now. Most engineers I know use at least one of them daily.",
         "tooling", hook=7, tone=8, x_algo=7, data_s=6, pillar_s=7, cta=6, nvl=False,
         reasoning="Opening is generic. Named tools but no specifics. Reasonable tone. Low debate potential. Tooling pillar present but weak. No CTA."),

    make("draft_0037",
         "Fine-tuning a small model can outperform prompting a large one for narrow tasks.\n\nWe saw this with a classification task: fine-tuned 7B outperformed GPT-4 zero-shot by 8 points.",
         "ai_research", hook=7, tone=8, x_algo=8, data_s=8, pillar_s=8, cta=6, nvl=False,
         reasoning="8-point improvement specific but task unnamed. Clean tone. Debate potential: depends on the task. Research pillar present. No CTA."),

    make("draft_0038",
         "The gap between what AI can do and what most businesses use it for is enormous.\n\nMost companies are still using it for chatbots and document summarization. The value is in decisions, not documents.",
         "takes", hook=7, tone=8, x_algo=8, data_s=5, pillar_s=8, cta=7, nvl=False,
         reasoning="No specific data for 'most companies'. Good take framing. Clean tone. Debate potential. Takes pillar. CTA: 'decisions not documents' is memorable."),

    make("draft_0039",
         "I switched from Python to Rust for our inference server.\n\nMemory allocations dropped by 60%. P99 latency went from 380ms to 140ms. The team learned Rust in 3 months.",
         "tooling", hook=8, tone=8, x_algo=8, data_s=9, pillar_s=8, cta=5, nvl=False,
         reasoning="60% allocation drop, latency numbers specific. Named language switch. Clean tone. Divisive: Rust migration debate. Tooling pillar. Weak CTA."),

    make("draft_0040",
         "Most AI demos work because they cherry-pick the inputs.\n\nThe real test is what happens on the 20th percentile input — the ugly, ambiguous, edge-case input that users will actually send.",
         "takes", hook=8, tone=8, x_algo=8, data_s=6, pillar_s=8, cta=7, nvl=False,
         reasoning="20th percentile is specific framing. Clean tone. Provokes product builders. Takes pillar. CTA: implicit challenge to test ugly inputs."),

    make("draft_0041",
         "Three signals that an ML team is in trouble:\n\n1. No staging environment for models\n2. Retraining is a manual process\n3. The eval metric is accuracy on the training set\n\nAll three are fixable in a sprint.",
         "career", hook=8, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=7, nvl=False,
         reasoning="Three specific signals named. Clean tone. Shareable diagnostic. Career/leadership pillar. CTA: 'fixable in a sprint' invites debate on difficulty."),

    make("draft_0042",
         "The attention mechanism was invented in 2014. It took 3 years for the transformer to use it properly.\n\nMost breakthroughs in AI are combinations of existing ideas, not new ideas. Read the old papers.",
         "ai_research", hook=8, tone=8, x_algo=8, data_s=8, pillar_s=8, cta=7, nvl=False,
         reasoning="2014/3 years are specific. Clean tone. Debate: what counts as a new idea. Research pillar. CTA: 'read old papers' is actionable."),

    make("draft_0043",
         "I've seen more ML projects fail from bad data than bad models.\n\nThe ratio in my experience: 70% data problems, 20% model problems, 10% infrastructure. People don't talk about this because data is unglamorous.",
         "tooling", hook=7, tone=8, x_algo=8, data_s=7, pillar_s=7, cta=6, nvl=False,
         reasoning="70/20/10 split is a specific claim but from personal experience. Clean tone. Debate: others will cite different ratios. Tooling pillar. Weak CTA."),

    make("draft_0044",
         "The best ML engineers I've worked with aren't the ones with the most ML knowledge.\n\nThey're the ones who know when to stop tweaking the model and fix the data pipeline.",
         "career", hook=7, tone=8, x_algo=8, data_s=5, pillar_s=8, cta=6, nvl=False,
         reasoning="Good framing but vague: no numbers. Clean tone. Career pillar. Reasonable debate potential. Weak CTA."),

    make("draft_0045",
         "Embeddings are one of the most useful primitives in ML.\n\nYou can use them for search, recommendations, clustering, anomaly detection, and classification — all with the same representation.",
         "tooling", hook=7, tone=8, x_algo=7, data_s=6, pillar_s=8, cta=5, nvl=False,
         reasoning="Useful information but not controversial. Clean tone. Low debate potential. Tooling pillar. No CTA."),

    make("draft_0046",
         "Hiring ML engineers in 2024 is broken.\n\nWe ask about neural network architecture in whiteboard interviews, then wonder why people who pass can't debug a data pipeline.\n\nThe interview tests the wrong skills.",
         "career", hook=8, tone=8, x_algo=8, data_s=6, pillar_s=9, cta=7, nvl=False,
         reasoning="Specific problem identified. Clean tone. High debate: interviewers will disagree. Career pillar strong. CTA: implicit challenge to redesign interviews."),

    make("draft_0047",
         "Transformers don't understand language. They predict tokens.\n\nThe distinction matters because it changes what you expect the model to do well on — and where it will fail in ways that look like understanding but aren't.",
         "ai_research", hook=8, tone=8, x_algo=8, data_s=6, pillar_s=8, cta=7, nvl=False,
         reasoning="Clean philosophical take. No specific numbers. Good debate bait. Research pillar. CTA: implicit challenge to anthropomorphize less."),

    make("draft_0048",
         "I turned down 3 ML consulting gigs this year.\n\nEach one wanted 'AI integration' but meant 'put a chatbot on our website.' The gap between what leaders think AI does and what it actually does is still the biggest problem in the industry.",
         "takes", hook=7, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=6, nvl=False,
         reasoning="3 gigs specific. Clean tone. Debate: chatbot dismissal will provoke product people. Takes pillar. Weak CTA."),

    make("draft_0049",
         "The difference between a 6-month ML project and a 2-year ML project:\n\nThe 6-month one defined the success metric before writing code.\nThe 2-year one discovered the metric was wrong 8 months in.",
         "career", hook=8, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=7, nvl=False,
         reasoning="Specific timeframes. Clean tone. Counterintuitive insight. Career pillar. CTA: implicit challenge to define metrics first."),

    make("draft_0050",
         "Claude 3 Opus and GPT-4 Turbo have similar coding performance on HumanEval.\n\nThe real differentiator: how they handle ambiguity. Claude asks clarifying questions. GPT-4 guesses and continues.\n\nNeither is universally better. Know which behavior your use case needs.",
         "tooling", hook=7, tone=8, x_algo=8, data_s=8, pillar_s=8, cta=7, nvl=False,
         reasoning="Named models, named benchmark. Clean tone. Useful comparison. Tooling pillar. CTA: 'know which behavior' is actionable."),

    make("draft_0051",
         "The ROI calculation for AI projects almost always ignores the human oversight cost.\n\nFor every AI output, someone has to review it. That review time is real labor. If your AI isn't 95%+ reliable, the oversight cost eats the savings.",
         "takes", hook=7, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=7, nvl=False,
         reasoning="95% threshold specific. Clean tone. Provokes CFOs and AI proponents. Takes pillar. CTA: implicit benchmark question."),

    make("draft_0052",
         "I switched our CI pipeline to use Claude for code review comments.\n\nIt catches 70% of the issues our human reviewers flag. It also flags 20% false positives.\n\nNot a replacement. A first pass that makes humans faster.",
         "tooling", hook=8, tone=8, x_algo=8, data_s=9, pillar_s=8, cta=7, nvl=False,
         reasoning="70% catch rate, 20% false positive rate specific. Named tool. Clean tone. Balanced view. Tooling pillar. CTA: implicit framing of AI as augmentation."),

    make("draft_0053",
         "You can read all the ML papers you want. The knowledge that compounds fastest is reading production postmortems.\n\nWhat failed in prod teaches you more in 20 minutes than a paper teaches you in 2 hours.",
         "career", hook=7, tone=8, x_algo=8, data_s=6, pillar_s=8, cta=7, nvl=False,
         reasoning="20 min vs 2 hours is specific. Clean tone. Contrarian view on learning sources. Career pillar. CTA: implicit action to find postmortems."),

    make("draft_0054",
         "Billion-parameter models are often the wrong tool.\n\nFor tabular data: gradient boosting still wins in most benchmarks.\nFor time series: classical statistical methods remain competitive.\n\nUse the right tool for the data type.",
         "ai_research", hook=7, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=7, nvl=False,
         reasoning="Named data types and methods. Clean tone. Contrarian: challenges LLM-for-everything trend. Research pillar. CTA: actionable framing."),

    make("draft_0055",
         "The best technical product managers I've worked with could read model evaluation reports without help.\n\nThe worst ones treated model quality as binary: 'does it work?' The difference in product quality those two approaches produce is enormous.",
         "career", hook=7, tone=8, x_algo=7, data_s=5, pillar_s=8, cta=6, nvl=False,
         reasoning="Good career observation. No specific numbers. Clean tone. Low debate potential. Career pillar. Weak CTA."),

    make("draft_0056",
         "Semantic search beats keyword search for queries with 5+ words.\n\nKeyword search beats semantic search for exact product names, codes, and IDs.\n\nHybrid search wins in practice because real user queries are a mix of both.",
         "tooling", hook=8, tone=8, x_algo=7, data_s=7, pillar_s=8, cta=6, nvl=False,
         reasoning="5+ words threshold specific. Named retrieval types. Clean tone. Moderate debate. Tooling pillar. Weak CTA."),

    make("draft_0057",
         "The career path from IC to staff ML engineer has one gate nobody talks about:\n\nYou have to stop being right about the model and start being right about the system.\n\nMost engineers stall here because model thinking doesn't transfer to systems thinking.",
         "career", hook=8, tone=8, x_algo=8, data_s=6, pillar_s=9, cta=7, nvl=False,
         reasoning="Clear framing of the IC-to-staff gap. Clean tone. Debate: what does 'systems thinking' mean. Career pillar strong. CTA: implicit challenge to grow."),

    make("draft_0058",
         "I ran a 90-day experiment: no AI coding tools, pure keyboard + docs.\n\nCode quality: unchanged. Speed on new features: -20%. Speed on refactoring: -35%.\n\nThe ROI on AI coding tools is highest for mechanical work, lowest for design.",
         "tooling", hook=8, tone=8, x_algo=8, data_s=9, pillar_s=8, cta=7, nvl=False,
         reasoning="90 days, -20%/-35% specific. Clean tone. Debate: others will dispute the numbers. Tooling pillar. CTA: 'mechanical vs design' framing."),

    make("draft_0059",
         "Every time a model gets a new capability, the benchmark for that capability immediately becomes saturated.\n\nThat's not progress failing. That's the benchmark being wrong. Better benchmarks, not better scores, measure real progress.",
         "ai_research", hook=8, tone=8, x_algo=8, data_s=5, pillar_s=8, cta=7, nvl=False,
         reasoning="Good meta-critique of benchmarks. No specific numbers. Clean tone. Debate: benchmark designers will push back. Research pillar. CTA: 'better benchmarks' is actionable."),

    make("draft_0060",
         "The phrase 'AI will take your job' is wrong in a predictable way.\n\nIt won't take the job. It will take the 40% of the job that you find tedious, and raise expectations for the other 60%.\n\nAdaptation is real. Replacement is mostly hype.",
         "takes", hook=8, tone=8, x_algo=8, data_s=6, pillar_s=8, cta=7, nvl=False,
         reasoning="40%/60% split is specific. Clean tone. Provokes both camps. Takes pillar clear. CTA: implicit challenge to hyped narratives."),

    make("draft_0061",
         "Deploying a model to production takes 3x longer than training it.\n\nData validation, schema versioning, load testing, rollback plans, monitoring dashboards — none of this is in the tutorial.\n\nIf your ML training is faster than your deploy, you're training too much.",
         "tooling", hook=8, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=8, nvl=False,
         reasoning="3x ratio specific. Named deploy components. Clean tone. Challenges tutorial culture. Tooling pillar. CTA: 'training too much' is a strong, debatable claim."),

    make("draft_0062",
         "The most important skill in ML engineering isn't deep learning theory.\n\nIt's knowing how to write a data pipeline that doesn't break when the upstream schema changes.\n\nSchemas change. Models overfit. Pipelines rot.",
         "career", hook=7, tone=8, x_algo=8, data_s=5, pillar_s=8, cta=7, nvl=False,
         reasoning="Clean professional take. No specific numbers. Good career insight. Moderate debate potential. Career pillar. CTA: 'pipelines rot' is memorable."),

    make("draft_0063",
         "I spent 3 weeks evaluating every major AI writing tool.\n\nRank by output quality for technical content: Claude 3 Opus, GPT-4, Gemini Ultra, Claude 3 Sonnet.\n\nThe gap between 1 and 4 is smaller than the marketing suggests.",
         "tooling", hook=7, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=7, nvl=False,
         reasoning="3 weeks, named tools in rank order. Clean tone. Debate: ranking will be disputed. Tooling pillar. CTA: 'gap is smaller' challenges marketing."),

    make("draft_0064",
         "The fastest path to production ML isn't a custom model.\n\nIt's identifying the one decision in your product that, if made 20% more accurately, would generate 5x more revenue — and building exactly that.\n\nNarrow problems, deployed fast, beat broad solutions in a drawer.",
         "takes", hook=8, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=8, nvl=False,
         reasoning="20% accuracy/5x revenue framing is specific. Clean tone. Provokes ML engineers building general solutions. Takes pillar. CTA: 'in a drawer' is memorable."),

    make("draft_0065",
         "Running LLMs locally has a hidden cost: your time.\n\nModel download: 40 minutes. Quantization troubleshooting: 2 hours. First working inference: 3 hours in.\n\nFor prototyping, the API is almost always faster to value.",
         "tooling", hook=7, tone=8, x_algo=8, data_s=8, pillar_s=8, cta=7, nvl=False,
         reasoning="40min/2hr/3hr specific. Clean tone. Debate: local LLM advocates will push back. Tooling pillar. CTA: 'faster to value' is a clear trade-off framing."),

    make("draft_0066",
         "Enterprise AI projects fail in a predictable order:\n\n1. The pilot worked because the data was curated\n2. Production data is messy\n3. The model needs retraining\n4. No one owns retraining\n5. Project dies\n\nOwnership beats everything.",
         "takes", hook=8, tone=8, x_algo=8, data_s=6, pillar_s=8, cta=8, nvl=False,
         reasoning="Clear numbered failure pattern. Clean tone. Highly shareable: enterprise ML failures are common. Takes pillar. CTA: 'ownership beats everything' is actionable."),

    make("draft_0067",
         "BERT was published in 2018. It still outperforms 60% of LLMs on sentence classification tasks.\n\nNewer isn't always better for narrow tasks with labeled data. Choose the oldest model that solves your problem.",
         "ai_research", hook=7, tone=8, x_algo=7, data_s=8, pillar_s=8, cta=7, nvl=False,
         reasoning="2018, 60% benchmark specific. Named model. Clean tone. Debate: LLM enthusiasts will challenge. Research pillar. CTA: 'oldest model that solves your problem' is actionable."),

    make("draft_0068",
         "The hardest technical interview question I ask:\n\n'How would you detect if your production model started performing worse without access to labels?'\n\nMost people with strong ML theory backgrounds can't answer this. It's a systems question dressed as an ML question.",
         "career", hook=8, tone=8, x_algo=8, data_s=7, pillar_s=8, cta=7, nvl=False,
         reasoning="Specific interview question. Clean tone. Invites people to share their answers. Career pillar. CTA: implicit challenge to answer."),

    make("draft_0069",
         "Speculative decoding can increase LLM throughput by 2-3x with no change in output quality.\n\nAlmost nobody uses it outside of research. The implementation complexity is the only barrier.\n\nGap between research and production is still enormous.",
         "ai_research", hook=8, tone=8, x_algo=8, data_s=8, pillar_s=8, cta=7, nvl=False,
         reasoning="2-3x throughput specific. Named technique. Clean tone. Debate: complexity vs. gain. Research pillar. CTA: implicit challenge to implement it."),

    make("draft_0070",
         "I've read 200 ML job descriptions this year. The skill listed most often: 'experience with LLMs.'\n\nThe skill mentioned least: 'experience debugging failed deploys.'\n\nThe market is hiring for the glamorous skill. The actual job is the unglamorous one.",
         "career", hook=8, tone=8, x_algo=8, data_s=8, pillar_s=8, cta=7, nvl=False,
         reasoning="200 job descriptions specific. Named gap. Clean tone. Debate: ML hiring process criticism. Career pillar. CTA: implicit challenge to hiring managers."),

    # ── FAILED_FLOOR no hashtag (composite < 8.0, never_list_violation=False) ─
    make("draft_0071",
         "AI is transforming every industry and unlocking new possibilities for businesses everywhere.",
         "takes", hook=2, tone=3, x_algo=2, data_s=1, pillar_s=3, cta=1, nvl=False,
         reasoning="Banned words: transforming, unlocking. No specifics. Vague opener. No CTA. Very low quality."),

    make("draft_0072",
         "Excited to share that I just finished my machine learning course! The content was amazing and I learned so much about neural networks and deep learning. Can't wait to apply these skills!",
         "career", hook=1, tone=2, x_algo=1, data_s=1, pillar_s=2, cta=1, nvl=False,
         reasoning="Exclamation marks. No specifics. Hedging tone. Weak hook. No debate potential. Low quality."),

    make("draft_0073",
         "The AI landscape is evolving rapidly and companies need to streamline their processes to stay competitive in this new ecosystem.",
         "takes", hook=1, tone=1, x_algo=1, data_s=1, pillar_s=2, cta=1, nvl=False,
         reasoning="Multiple banned words: landscape, streamline, ecosystem. Generic. No specifics. Very low quality."),

    make("draft_0074",
         "Just deployed my first ML model to production — it was a great experience and I'm so grateful for the amazing team I got to work with!",
         "tooling", hook=1, tone=2, x_algo=1, data_s=1, pillar_s=2, cta=1, nvl=False,
         reasoning="Exclamation mark. No specifics. Vague. No debate potential. Weak hook. Low quality."),

    make("draft_0075",
         "AI tools can really help streamline your workflow and make you more productive. Have you tried using them? What's your experience been like?",
         "tooling", hook=2, tone=2, x_algo=2, data_s=1, pillar_s=2, cta=3, nvl=False,
         reasoning="Banned word: streamline. Soft CTA: 'What's your experience'. No specifics. Vague hook. Low quality."),

    make("draft_0076",
         "Thoughts on the future of AI? I think it's going to be transformative for how we work and live. Would love to hear your perspectives on this important topic.",
         "takes", hook=2, tone=2, x_algo=2, data_s=1, pillar_s=2, cta=2, nvl=False,
         reasoning="Banned word: transformative. Soft CTA: 'Would love to hear'. No specifics. Very generic."),

    make("draft_0077",
         "I built a chatbot last week. It uses GPT-4. It works pretty well for our use case. Let me know what you think about chatbots in general.",
         "tooling", hook=3, tone=5, x_algo=3, data_s=2, pillar_s=4, cta=3, nvl=False,
         reasoning="No specifics beyond 'GPT-4'. Soft CTA. Low hook quality. Vague result. Below average."),

    make("draft_0078",
         "Machine learning is a game-changer for data-driven companies looking to unlock the power of their data assets.",
         "takes", hook=1, tone=1, x_algo=1, data_s=1, pillar_s=1, cta=1, nvl=False,
         reasoning="Multiple banned words: game-changer, unlock. Extremely generic. No specifics. Lowest quality."),

    make("draft_0079",
         "Share your thoughts: what AI tool has had the biggest impact on your work this year?",
         "tooling", hook=2, tone=3, x_algo=2, data_s=1, pillar_s=2, cta=2, nvl=False,
         reasoning="Soft CTA: 'Share your thoughts'. No insight or hook. Just a question. Low quality."),

    make("draft_0080",
         "Learning Python for ML? Start with NumPy and Pandas before jumping to PyTorch. The fundamentals matter — you'll thank yourself later.",
         "career", hook=4, tone=5, x_algo=4, data_s=3, pillar_s=5, cta=4, nvl=False,
         reasoning="Em-dash used. Reasonable advice but vague. No specifics. Moderate quality but below threshold."),

    make("draft_0081",
         "Just finished reading the Attention Is All You Need paper. Mind blown! The way transformers work is honestly genius.",
         "ai_research", hook=2, tone=2, x_algo=2, data_s=2, pillar_s=3, cta=1, nvl=False,
         reasoning="Exclamation mark. No original insight. Vague praise. No specifics beyond paper name. Low quality."),

    make("draft_0082",
         "The AI revolution is here and it's transforming the technology landscape faster than anyone expected. What does this mean for your business?",
         "takes", hook=1, tone=1, x_algo=1, data_s=1, pillar_s=1, cta=2, nvl=False,
         reasoning="Banned words: transforming, landscape. Soft CTA. No specifics. Very generic. Lowest quality."),

    make("draft_0083",
         "I think vector databases are pretty useful. Pinecone and Weaviate are two good options. Both have free tiers if you want to try them out.",
         "tooling", hook=3, tone=6, x_algo=3, data_s=3, pillar_s=5, cta=3, nvl=False,
         reasoning="No specifics on performance. No insight. Named tools but no comparison. Soft suggestion. Below threshold."),

    make("draft_0084",
         "Struggling with model overfitting? Here are my top 5 tips to help you overcome this challenge in your ML projects!",
         "tooling", hook=3, tone=2, x_algo=2, data_s=2, pillar_s=4, cta=3, nvl=False,
         reasoning="Exclamation mark. Clickbait hook. No specifics. Generic tips format. Low quality."),

    make("draft_0085",
         "AI is genuinely exciting — the pace of progress is incredible and I'm fascinated by where this is all going.",
         "takes", hook=2, tone=2, x_algo=1, data_s=1, pillar_s=2, cta=1, nvl=False,
         reasoning="Em-dash. Exclamation in 'incredible'. No specifics. Vague enthusiasm. Very low quality."),

    # ── HASHTAG examples (never_list_violation=True, quality_tier=failed_floor) ─
    make("draft_0086",
         "Just launched our new AI feature! #MachineLearning #AI #ProductLaunch",
         "tooling", hook=1, tone=5, x_algo=3, data_s=1, pillar_s=2, cta=2, nvl=True,
         reasoning="Hashtags present: never_list_violation. No content. Announcement without substance."),

    make("draft_0087",
         "Great results from our model training run today. Accuracy improved significantly. #ML #DeepLearning #AI",
         "ai_research", hook=2, tone=5, x_algo=3, data_s=2, pillar_s=3, cta=1, nvl=True,
         reasoning="Hashtags present: never_list_violation. Vague 'significantly improved'. No specifics."),

    make("draft_0088",
         "The future of work is AI-powered! Exciting times ahead for everyone in the tech industry. #FutureOfWork #ArtificialIntelligence",
         "takes", hook=1, tone=3, x_algo=2, data_s=1, pillar_s=1, cta=1, nvl=True,
         reasoning="Hashtags present: never_list_violation. Exclamation. Generic enthusiasm. Banned framing."),

    make("draft_0089",
         "Check out this amazing paper on transformer attention that just dropped! #NLP #Transformers #Research",
         "ai_research", hook=2, tone=3, x_algo=2, data_s=1, pillar_s=3, cta=2, nvl=True,
         reasoning="Hashtags present: never_list_violation. No specifics on paper. Exclamation. Low quality."),

    make("draft_0090",
         "New blog post: 10 ways AI can help your startup grow faster! #Startup #AI #Growth #Entrepreneurship",
         "takes", hook=2, tone=2, x_algo=2, data_s=1, pillar_s=2, cta=3, nvl=True,
         reasoning="Hashtags present: never_list_violation. Exclamation. Clickbait format. No substance."),

    make("draft_0091",
         "Hiring ML engineers! Our team is growing and we're looking for passionate people who love building AI products. DM me if interested. #Hiring #MLJobs #AI",
         "career", hook=2, tone=4, x_algo=3, data_s=2, pillar_s=3, cta=4, nvl=True,
         reasoning="Hashtags present: never_list_violation. Soft CTA: 'DM me'. No specifics on role or tech stack."),

    make("draft_0092",
         "Just passed my AWS ML Specialty certification! Hard work pays off. #AWS #MachineLearning #CloudComputing #Certification",
         "career", hook=2, tone=4, x_algo=2, data_s=2, pillar_s=3, cta=1, nvl=True,
         reasoning="Hashtags present: never_list_violation. Exclamation. No insight. Personal milestone without lesson."),

    make("draft_0093",
         "LLMs are revolutionizing enterprise software! The game-changer no one saw coming. #LLM #Enterprise #AIRevolution",
         "takes", hook=2, tone=1, x_algo=2, data_s=1, pillar_s=2, cta=1, nvl=True,
         reasoning="Hashtags present: never_list_violation. Exclamation. Banned words: revolutionizing, game-changer. Very low quality."),

    make("draft_0094",
         "Grateful for this community! The discussions here always inspire me to keep learning and growing. #AI #Community #Learning",
         "takes", hook=1, tone=4, x_algo=1, data_s=1, pillar_s=1, cta=1, nvl=True,
         reasoning="Hashtags present: never_list_violation. Exclamation. No substance. Community-bait post. Very low quality."),

    make("draft_0095",
         "My take: #RAG is overrated for most use cases. Fine-tuning gets you further if your domain is specialized enough.",
         "tooling", hook=4, tone=5, x_algo=4, data_s=4, pillar_s=5, cta=4, nvl=True,
         reasoning="Hashtag in content: never_list_violation. Decent take but disqualified by hashtag usage."),

    make("draft_0096",
         "Morning motivation for all my fellow #MLEngineers: the only way to learn is by shipping things that break. Keep building!",
         "career", hook=2, tone=3, x_algo=2, data_s=1, pillar_s=2, cta=2, nvl=True,
         reasoning="Hashtag present: never_list_violation. Generic motivation. No specifics. Low quality."),

    make("draft_0097",
         "Is RAG dead? Some are saying #VectorSearch is being replaced by long-context models. What do you think?",
         "ai_research", hook=5, tone=5, x_algo=5, data_s=3, pillar_s=5, cta=4, nvl=True,
         reasoning="Hashtag present: never_list_violation. Soft CTA: 'What do you think'. Despite interesting question, hashtag disqualifies."),

    make("draft_0098",
         "We're at #NeurIPS2024 this week. Exciting papers on multi-modal models and long-context transformers. Will share highlights!",
         "ai_research", hook=3, tone=5, x_algo=3, data_s=3, pillar_s=4, cta=3, nvl=True,
         reasoning="Hashtag present: never_list_violation. Conference announcement. No substance yet. Low quality."),

    make("draft_0099",
         "Nobody talks about the mental health cost of being in AI right now. The pace is brutal. Burnout is real. #AIBurnout #TechWellness",
         "takes", hook=5, tone=5, x_algo=4, data_s=3, pillar_s=5, cta=3, nvl=True,
         reasoning="Hashtags present: never_list_violation. Interesting topic but disqualified. No specifics."),

    make("draft_0100",
         "Dropped a new tutorial on fine-tuning Llama 3 with QLoRA! Link in bio. #LLM #FineTuning #Llama #OpenSource",
         "tooling", hook=3, tone=4, x_algo=3, data_s=2, pillar_s=4, cta=3, nvl=True,
         reasoning="Hashtags present: never_list_violation. Link-drop CTA. No substance in tweet itself."),
]

OUT = Path(__file__).parent / "batch_01.jsonl"
with OUT.open("w", encoding="utf-8") as f:
    for ex in EXAMPLES:
        f.write(json.dumps(ex) + "\n")

print(f"Written {len(EXAMPLES)} examples to {OUT}")
