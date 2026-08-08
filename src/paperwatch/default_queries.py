from __future__ import annotations

from typing import Any


CLASSIFIER_OUTPUT = (
    "For each record, select exactly one allowed value. First line: "
    "RESULT: <value>. Second line: WHY: <one sentence citing source evidence>. "
    "Use Unknown or Unclear when the source does not contain enough evidence. "
    "Do not infer from reputation or author identity alone."
)


def _labels(*values: str) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "value": value,
            "pattern": rf"(?im)^\s*RESULT:\s*{value}\s*$",
        }
        for value in values
    )


def _classifier(
    name: str,
    label_class: str,
    instruction: str,
    *values: str,
    content_scope: str = "papers",
    paper_source_scope: str = "fulltext",
) -> dict[str, Any]:
    allowed_values = tuple(values) if any(value.upper() == "STUB" for value in values) else (*values, "STUB")
    return {
        "name": name,
        "prompt": f"{instruction}\n\nAllowed values: {', '.join(allowed_values)}.\n\n{CLASSIFIER_OUTPUT}",
        "content_scope": content_scope,
        "paper_source_scope": paper_source_scope,
        "pdf_strategy": "chunked",
        "query_kind": "classification",
        "label_class": label_class,
        "label_values": _labels(*allowed_values),
    }


def _analysis(
    name: str,
    prompt: str,
    *,
    content_scope: str = "papers",
    paper_source_scope: str = "fulltext",
) -> dict[str, Any]:
    return {
        "name": name,
        "prompt": prompt,
        "content_scope": content_scope,
        "paper_source_scope": paper_source_scope,
        "pdf_strategy": "chunked",
        "query_kind": "summary",
    }


def _regexp(
    name: str,
    pattern: str,
    radius: int,
    *,
    content_scope: str = "both",
) -> dict[str, Any]:
    return {
        "name": name,
        "prompt": "",
        "content_scope": content_scope,
        "paper_source_scope": "auto",
        "pdf_strategy": "auto",
        "regexp_pattern": pattern,
        "regexp_radius": radius,
        "regexp_mode": "print",
        "query_kind": "summary",
    }


DEFAULT_SAVED_QUERIES: tuple[dict[str, Any], ...] = (
    _classifier(
        "Dominant contribution type",
        "Contribution",
        "Classify the paper by its primary intellectual contribution, not merely the techniques it uses.",
        "Theory",
        "Method",
        "Systems",
        "Dataset Benchmark",
        "Empirical Study",
        "Survey Position",
        "Replication",
        "Negative Result",
        "Other",
        "Unknown",
    ),
    _classifier(
        "Research maturity",
        "Maturity",
        "Classify the most advanced stage actually demonstrated. Do not treat a proposed architecture or simulated result as a working prototype.",
        "Conceptual",
        "Simulation",
        "Lab Prototype",
        "Field Evaluated",
        "Deployed",
        "Not Applicable",
        "Unknown",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _classifier(
        "Evidence strength",
        "Evidence",
        "Judge the strongest support for the central claim, considering controls, baselines, sample size, proofs, ablations, independent datasets, and external validity.",
        "Formal Proof",
        "Strong Empirical",
        "Moderate Empirical",
        "Preliminary",
        "Position Speculation",
        "Mixed",
        "Unknown",
    ),
    _classifier(
        "Novelty level",
        "Novelty",
        "Judge what is technically new relative to the closest prior work described in the source. Separate new combinations and scale-ups from new mechanisms or results.",
        "High",
        "Substantial",
        "Incremental",
        "Replication",
        "Negative Result",
        "Unclear",
    ),
    _classifier(
        "Artifact openness",
        "Artifacts",
        "Classify only explicitly available research artifacts. Planned releases do not count as open. Distinguish code, data, weights, and sufficient reproduction instructions.",
        "Fully Open",
        "Open Code",
        "Open Data",
        "Open Weights",
        "Partial",
        "Planned",
        "None",
        "Unknown",
    ),
    _classifier(
        "Institutional backing",
        "Backing",
        "Use affiliations, acknowledgements, funding statements, grant identifiers, contracts, and conflict disclosures. Classify the dominant backing of the work, not the publication venue.",
        "Government",
        "Public Research",
        "Private Industry",
        "Nonprofit",
        "Mixed",
        "Independent",
        "Unknown",
    ),
    _classifier(
        "Military and dual-use relevance",
        "Dual Use",
        "Distinguish explicit defense involvement from technically plausible dual use. Consider surveillance, targeting, autonomous systems, intelligence analysis, cyber operations, logistics, communications, sensing, and weapons development. Do not label generic computing research without a concrete pathway.",
        "Explicit Military",
        "Direct Dual Use",
        "Indirect Dual Use",
        "No Evident Relevance",
        "Unclear",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _classifier(
        "Deployment horizon",
        "Horizon",
        "Estimate how close the demonstrated result is to practical use, accounting for missing engineering, data access, regulation, cost, reliability, and integration rather than author optimism.",
        "Deployable Now",
        "Near Term",
        "Medium Term",
        "Long Term",
        "Foundational",
        "Unclear",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _classifier(
        "Compute intensity",
        "Compute",
        "Classify the resources needed to reproduce or use the central result relative to a typical academic research group. Consider training, inference, specialized hardware, memory, wall-clock time, and proprietary frontier infrastructure.",
        "Low",
        "Moderate",
        "High",
        "Frontier Scale",
        "Not Applicable",
        "Unknown",
    ),
    _classifier(
        "Topic applicability - edit focus topic",
        "Topic Fit",
        "The focus topic is: [REPLACE WITH FOCUS TOPIC]. Judge whether the record directly advances this topic, supplies an important enabling technique, is merely adjacent, or is unrelated. Evaluate mechanisms and results, not keyword occurrence.",
        "Direct",
        "Enabling",
        "Adjacent",
        "Unrelated",
        "Unknown",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _analysis(
        "Frontier significance",
        "Explain why this work matters at the research frontier. Identify what changed relative to prior capabilities, why the result became possible now, the strongest supporting evidence, the most likely downstream consequences, and the finding that would most weaken its significance. Separate genuine technical leverage from scale, packaging, or benchmark optimization.",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _analysis(
        "Novelty audit",
        "Audit the paper's novelty. Identify the closest prior approaches named in the paper, what is unchanged, what is newly combined, and what is genuinely new. Separate conceptual novelty, technical novelty, empirical novelty, and scale effects. Highlight novelty claims not adequately supported by comparison or ablation evidence.",
    ),
    _analysis(
        "Claim-evidence audit",
        "List the paper's major claims and the evidence supporting each one. For every claim, identify the proof, experiment, benchmark, ablation, case study, or argument used; then assess whether that evidence fully supports the claim. Highlight missing controls, weak baselines, confounders, underpowered evaluations, unsupported generalization, and claim-evidence mismatches.",
    ),
    _analysis(
        "Applicability and blockers",
        "Map the practical applicability of this work. Identify likely users, suitable environments, prerequisites, required data and infrastructure, integration points, economic or regulatory constraints, and failure conditions. Distinguish what can be used immediately from what requires further research or engineering. Give concrete application examples and the principal blocker for each.",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _analysis(
        "Reproducibility brief",
        "Produce a reproduction assessment. Record available code, data, weights, licenses, environment details, hyperparameters, compute requirements, evaluation procedures, and randomization details. Identify missing information and hidden dependencies. End with a minimal reproduction plan, estimated resource category, and the three most likely reasons an independent reproduction would fail.",
    ),
    _analysis(
        "Implementation brief",
        "Convert the work into an implementation-oriented technical brief. Describe inputs, outputs, major components, data flow, training or optimization procedure, inference procedure, dependencies, key equations, and computational bottlenecks. Provide concise pseudocode or an ordered workflow when possible. Distinguish essential mechanisms from optional refinements and evaluation-only components.",
    ),
    _analysis(
        "Military and dual-use assessment",
        "Assess military, intelligence, surveillance, cyber, and other dual-use implications without sensationalism. Identify explicit defense links separately from inferred applications. For each plausible use, describe the capability enabled, required adaptations, current technical barriers, likely actors, and misuse potential. Include benign applications of the same capability and assign a confidence level to every inference.",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _analysis(
        "Funding and provenance report",
        "Extract the institutional provenance of the work: author affiliations, laboratories, funders, grant or contract identifiers, corporate involvement, government involvement, donated infrastructure, conflicts of interest, and declared sponsors. Classify each item as government, public research, private industry, nonprofit, or unclear. Quote or precisely paraphrase the supporting source text and do not infer funding from affiliation alone.",
    ),
    _analysis(
        "Research landscape map",
        "Place the work in its research landscape using only references and comparisons present in the source. Identify foundational work it builds on, its closest competitors, alternative research directions, and the papers most useful to read next. Explain the relationship to each cited work rather than producing a citation list. Highlight important approaches the authors mention but do not evaluate.",
    ),
    _analysis(
        "Next experiments and open problems",
        "Propose the highest-information follow-up work grounded in the source. Include decisive experiments, missing ablations, stress tests, external-validity tests, theoretical questions, scaling tests, and replication targets. Rank them by expected information gain and feasibility. For each, state the uncertainty it resolves, required resources, expected outcomes, and what a negative result would imply.",
        content_scope="both",
        paper_source_scope="auto",
    ),
    _regexp(
        "Funding and grants",
        r"(?i)\b(?:funded\s+by|financial\s+support\s+(?:from|by)|sponsored\s+by|grant\s+(?:no\.?|number)|contract\s+(?:no\.?|number)|award\s+(?:no\.?|number))\b",
        100,
    ),
    _regexp(
        "Defense and military references",
        r"(?i)\b(?:military|defen[cs]e|warfighting|battlefield|weapon(?:s|ry)?|dual[- ]use|DARPA|IARPA|AFOSR|ONR|Army Research (?:Office|Laboratory)|Naval Research Laboratory|Department of Defense|Ministry of Defen[cs]e)\b",
        150,
    ),
    _regexp(
        "Source code and repositories",
        r"(?i)\b(?:source\s+code|code\s+(?:is|will be)\s+(?:available|released)|implementation\s+(?:is|will be)\s+(?:available|released))\b|https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/\S+",
        50,
    ),
    _regexp(
        "Data, weights, and research artifacts",
        r"(?i)\b(?:dataset|data|model weights?|checkpoints?)\s+(?:is|are|will be)\s+(?:publicly\s+)?(?:available|released)\b|https?://(?:huggingface\.co|zenodo\.org|figshare\.com|osf\.io)/\S+",
        60,
    ),
    _regexp(
        "Compute and hardware requirements",
        r"(?i)\b(?:GPU[- ]?hours?|TPU[- ]?hours?|FLOPs?|petaflop|exaflop|A100|H100|H200|B100|B200|TPU\s*v\d|(?:billion|million)\s+parameters?|parameter\s+count|training\s+compute)\b",
        100,
    ),
    _regexp(
        "Limitations and impact sections",
        r"(?im)^\s*(?:\d+(?:\.\d+)*[\s.: -]*)?(?:limitations?|threats?\s+to\s+validity|failure\s+modes?|open\s+questions?|broader\s+impacts?)\s*$",
        350,
        content_scope="papers",
    ),
    _regexp(
        "Real-world deployment evidence",
        r"(?i)\b(?:deployed\s+in|production\s+(?:system|environment|traffic)|field\s+(?:trial|study|test)|real[- ]world\s+(?:deployment|evaluation)|prospective\s+(?:study|trial)|in[- ]the[- ]wild)\b",
        150,
    ),
    _regexp(
        "Statistical uncertainty and rigor",
        r"(?i)\b(?:confidence\s+interval|credible\s+interval|standard\s+error|statistically\s+significant|p[- ]?value|effect\s+size|multiple\s+comparisons?|power\s+analysis)\b",
        80,
        content_scope="papers",
    ),
    _regexp(
        "Strong novelty and performance claims",
        r"(?i)\b(?:state[- ]of[- ]the[- ]art|SOTA|outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|first\s+to|for\s+the\s+first\s+time|significant(?:ly)?\s+improv(?:e|es|ed|ement))\b",
        120,
    ),
    _regexp(
        "Failures, negative results, and impossibility",
        r"(?i)\b(?:negative\s+result|failed\s+to|fails\s+to|does\s+not\s+outperform|no\s+significant\s+(?:difference|improvement)|performance\s+degradation|counterexample|impossibility\s+result|lower\s+bound)\b",
        150,
    ),
)


def add_default_queries(db: Any, model_profile: str = "") -> int:
    """Add missing optional defaults without enabling automatic execution."""
    existing = {str(row["name"]).casefold() for row in db.list_saved_queries()}
    added = 0
    for definition in DEFAULT_SAVED_QUERIES:
        if definition["name"].casefold() in existing:
            continue
        values = dict(definition)
        values["model_profiles"] = ()
        values["auto_run_papers"] = False
        values["auto_run_blogs"] = False
        db.create_saved_query(**values)
        existing.add(definition["name"].casefold())
        added += 1
    return added
