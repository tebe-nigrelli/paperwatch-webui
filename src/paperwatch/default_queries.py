from __future__ import annotations

from typing import Any

from .db import utcnow


CLASSIFIER_OUTPUT = (
    "For each record, select exactly one allowed value. First line: "
    "RESULT: <value>. Second line: WHY: <one sentence citing source evidence>. "
    "Use Unknown or Unclear when the source does not contain enough evidence. "
    "Do not infer from reputation or author identity alone."
)


LEGACY_REGEXP_PATTERNS: dict[str, str] = {
    "Funding and grants": r"(?i)\b(?:funded\s+by|financial\s+support\s+(?:from|by)|sponsored\s+by|grant\s+(?:no\.?|number)|contract\s+(?:no\.?|number)|award\s+(?:no\.?|number))\b",
    "Defense and military references": r"(?i)\b(?:military|defen[cs]e|warfighting|battlefield|weapon(?:s|ry)?|dual[- ]use|DARPA|IARPA|AFOSR|ONR|Army Research (?:Office|Laboratory)|Naval Research Laboratory|Department of Defense|Ministry of Defen[cs]e)\b",
    "Source code and repositories": r"(?i)\b(?:source\s+code|code\s+(?:is|will be)\s+(?:available|released)|implementation\s+(?:is|will be)\s+(?:available|released))\b|https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/\S+",
    "Data, weights, and research artifacts": r"(?i)\b(?:dataset|data|model weights?|checkpoints?)\s+(?:is|are|will be)\s+(?:publicly\s+)?(?:available|released)\b|https?://(?:huggingface\.co|zenodo\.org|figshare\.com|osf\.io)/\S+",
    "Compute and hardware requirements": r"(?i)\b(?:GPU[- ]?hours?|TPU[- ]?hours?|FLOPs?|petaflop|exaflop|A100|H100|H200|B100|B200|TPU\s*v\d|(?:billion|million)\s+parameters?|parameter\s+count|training\s+compute)\b",
    "Limitations and impact sections": r"(?im)^\s*(?:\d+(?:\.\d+)*[\s.: -]*)?(?:limitations?|threats?\s+to\s+validity|failure\s+modes?|open\s+questions?|broader\s+impacts?)\s*$",
    "Real-world deployment evidence": r"(?i)\b(?:deployed\s+in|production\s+(?:system|environment|traffic)|field\s+(?:trial|study|test)|real[- ]world\s+(?:deployment|evaluation)|prospective\s+(?:study|trial)|in[- ]the[- ]wild)\b",
    "Statistical uncertainty and rigor": r"(?i)\b(?:confidence\s+interval|credible\s+interval|standard\s+error|statistically\s+significant|p[- ]?value|effect\s+size|multiple\s+comparisons?|power\s+analysis)\b",
    "Strong novelty and performance claims": r"(?i)\b(?:state[- ]of[- ]the[- ]art|SOTA|outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|first\s+to|for\s+the\s+first\s+time|significant(?:ly)?\s+improv(?:e|es|ed|ement))\b",
    "Failures, negative results, and impossibility": r"(?i)\b(?:negative\s+result|failed\s+to|fails\s+to|does\s+not\s+outperform|no\s+significant\s+(?:difference|improvement)|performance\s+degradation|counterexample|impossibility\s+result|lower\s+bound)\b",
}


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
        r"(?i)\b(?:funded\s+by|funding\s+(?:from|provided\s+by)|financial\s+support\s+(?:from|by)|supported\s+by\s+(?:the\s+)?(?:NSF|NIH|ERC|UKRI|DOE|DARPA|IARPA)|sponsored\s+by|under\s+(?:grant|contract)|grant\s+(?:agreement|no\.?|number)|contract\s+(?:no\.?|number)|award\s+(?:no\.?|number)|NSF\s+grant|NIH\s+grant|ERC\s+grant)\b",
        100,
    ),
    _regexp(
        "Defense and military references",
        r"(?i)\b(?:military|defen[cs]e|warfighting|battlefield|weapon(?:s|ry)?|dual[- ]use|DARPA|IARPA|AFOSR|ONR|DoD|Department of Defense|Ministry of Defen[cs]e|Army Research (?:Office|Laboratory)|Naval Research Laboratory|Air Force Research Laboratory|USAF|U\.S\.\s+Army|U\.S\.\s+Navy|NATO)\b",
        150,
    ),
    _regexp(
        "Source code and repositories",
        r"(?i)\b(?:source\s+code|open[- ]source(?:d)?|code\s+(?:is|will be|has been)\s+(?:publicly\s+)?(?:available|released)|implementation\s+(?:is|will be|has been)\s+(?:publicly\s+)?(?:available|released)|code\s+available\s+at)\b|\b(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/[^\s<>\"')\]]+",
        50,
    ),
    _regexp(
        "Data, weights, and research artifacts",
        r"(?i)\b(?:(?:dataset|data|model weights?|pretrained weights?|checkpoints?|research artifacts?)\s+(?:is|are|will be|have been)\s+(?:publicly\s+)?(?:available|released)|release(?:d|s|ing)?\s+(?:the\s+)?(?:dataset|model weights?|checkpoints?)|artifact\s+(?:repository|release))\b|\b(?:www\.)?(?:huggingface\.co|zenodo\.org|figshare\.com|osf\.io)/[^\s<>\"')\]]+",
        60,
    ),
    _regexp(
        "Compute and hardware requirements",
        r"(?i)\b(?:GPU[- ]?(?:hours?|days?)|TPU[- ]?(?:hours?|days?)|core[- ]hours?|node[- ]hours?|FLOPs?|TFLOPs?|PFLOPs?|petaflop|exaflop|A100|H100|H200|B100|B200|V100|L40S|MI250|MI300X|RTX\s*(?:3090|4090|5090)|DGX(?:\s+\w+)?|TPU\s*v\d(?:e|p)?|GPU\s+memory|VRAM|(?:billion|million)\s+parameters?|parameter\s+count|training\s+compute|training\s+time|inference\s+latency|tokens?\s*/\s*s(?:ec(?:ond)?)?)\b",
        100,
    ),
    _regexp(
        "Limitations and impact sections",
        r"(?im)^\s*(?:\d+(?:\.\d+)*[\s.: -]*)?(?:limitations?(?:\s+and\s+future\s+work)?|threats?\s+to\s+validity|failure\s+modes?|open\s+questions?|broader\s+impacts?|societal\s+impacts?|ethical\s+considerations|risks?\s+and\s+limitations?)\s*$",
        350,
        content_scope="papers",
    ),
    _regexp(
        "Real-world deployment evidence",
        r"(?i)\b(?:deployed\s+(?:in|at|to)|production\s+(?:system|environment|traffic|deployment)|in\s+production|live\s+traffic|operational\s+deployment|pilot\s+deployment|field\s+(?:trial|study|test|deployment)|real[- ]world\s+(?:deployment|evaluation|trial)|prospective\s+(?:study|trial)|clinical\s+trial|in[- ]the[- ]wild)\b",
        150,
    ),
    _regexp(
        "Statistical uncertainty and rigor",
        r"(?i)\b(?:confidence\s+interval|credible\s+interval|95%\s+CI|standard\s+error|statistically\s+significant|statistical\s+significance|p[- ]?value|effect\s+size|Cohen'?s\s+d|multiple\s+comparisons?|power\s+analysis|Bonferroni|Benjamini[- ]Hochberg|bootstrap(?:ped)?\s+confidence)\b",
        80,
        content_scope="papers",
    ),
    _regexp(
        "Strong novelty and performance claims",
        r"(?i)\b(?:state[- ]of[- ]the[- ]art|SOTA|new\s+state[- ]of[- ]the[- ]art|outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|improves?\s+over\s+(?:the\s+)?(?:baseline|prior\s+work)|beats?\s+(?:the\s+)?baseline|record\s+performance|first\s+to|first\s+demonstration|for\s+the\s+first\s+time|significant(?:ly)?\s+improv(?:e|es|ed|ement))\b",
        120,
    ),
    _regexp(
        "Failures, negative results, and impossibility",
        r"(?i)\b(?:negative\s+result|failed\s+to|fails\s+to|failure\s+case|does\s+not\s+outperform|underperform(?:s|ed|ing)?|no\s+improvement|no\s+significant\s+(?:difference|improvement)|performance\s+degradation|degrades?\s+performance|counterexample|impossibility\s+result|provably\s+impossible|lower\s+bound)\b",
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


def refresh_default_regexp_patterns(db: Any) -> int:
    """Upgrade untouched bundled regexp defaults without overwriting user edits."""
    current = {
        str(item["name"]): str(item["regexp_pattern"])
        for item in DEFAULT_SAVED_QUERIES
        if item.get("regexp_mode") == "print"
    }
    updated = 0
    with db.transaction() as conn:
        for name, old_pattern in LEGACY_REGEXP_PATTERNS.items():
            new_pattern = current.get(name)
            if not new_pattern or new_pattern == old_pattern:
                continue
            cur = conn.execute(
                """
                UPDATE saved_queries
                SET regexp_pattern=?, updated_at=?
                WHERE name=? AND regexp_mode='print' AND regexp_pattern=?
                """,
                (new_pattern, utcnow(), name, old_pattern),
            )
            updated += cur.rowcount
    return updated
