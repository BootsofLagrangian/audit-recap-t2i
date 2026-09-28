"use strict";

const app = document.querySelector("#app");
const tutorialAnnouncer = document.querySelector("#tutorial-announcer");
const tutorialStorageKey = "human-cbu-interface-tutorial-v2";
const participantCodeStorageKey = "human-cbu-participant-code-v1";
const state = {
  token: sessionStorage.getItem("human-cbu-session"),
  bootstrap: null,
  task: null,
  answers: {},
  startedAt: null,
  imagePrimaryLocked: false,
  imageRevealed: false,
  coverageCaption: null,
  imageObjectUrl: null,
  consentVersion: localStorage.getItem("human-cbu-consent-version"),
  participantInformation: null,
  withdrawing: false,
  practiceMode: false,
  testMode: false,
  previousTask: null,
  reviewingPrevious: false,
  pairTransition: false,
  tutorialStep: -1,
  tutorialAnswers: {},
  tutorialFeedback: "",
  tutorialTask: null,
  tutorialProgress: null,
  tutorialReturnStep: null,
  captionCategoryDraft: null,
  guideStartedAt: null,
  guideElapsedMs: 0,
};

const icons = {
  lock: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
    <rect x="5" y="10" width="14" height="10" rx="2"></rect><path d="M8 10V7a4 4 0 0 1 8 0v3"></path>
  </svg>`,
  arrow: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
    <path d="M5 12h14M13 6l6 6-6 6"></path>
  </svg>`,
  info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
    <circle cx="12" cy="12" r="9"></circle><path d="M12 11v6M12 7.5h.01"></path>
  </svg>`,
  zoom: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
    <circle cx="10.5" cy="10.5" r="6.5"></circle><path d="m15.5 15.5 5 5M10.5 7.5v6M7.5 10.5h6"></path>
  </svg>`,
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Content-Type", "application/json");
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const response = await fetch(path, {...options, headers});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderHighlightRanges(text, ranges, className = "") {
  let cursor = 0;
  const chunks = [];
  for (const range of ranges) {
    chunks.push(escapeHtml(text.slice(cursor, range.start)));
    chunks.push(`<mark${className ? ` class="${className}"` : ""}>${escapeHtml(text.slice(range.start, range.end))}</mark>`);
    cursor = range.end;
  }
  chunks.push(escapeHtml(text.slice(cursor)));
  return chunks.join("");
}

function lexicalTokens(text) {
  return [...String(text || "").matchAll(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu)].map((match) => ({
    value: match[0].toLocaleLowerCase(),
    start: match.index,
    end: match.index + match[0].length,
  }));
}

function lexicalCueRanges(caption, unit) {
  const captionTokens = lexicalTokens(caption);
  const unitTokens = lexicalTokens(unit);
  if (!captionTokens.length || !unitTokens.length) return [];

  let best = null;
  for (let startIndex = 0; startIndex < captionTokens.length; startIndex += 1) {
    if (captionTokens[startIndex].value !== unitTokens[0].value) continue;
    const ranges = [captionTokens[startIndex]];
    let captionIndex = startIndex + 1;
    let matched = true;
    for (let unitIndex = 1; unitIndex < unitTokens.length; unitIndex += 1) {
      while (
        captionIndex < captionTokens.length &&
        captionTokens[captionIndex].value !== unitTokens[unitIndex].value
      ) {
        captionIndex += 1;
      }
      if (captionIndex >= captionTokens.length) {
        matched = false;
        break;
      }
      ranges.push(captionTokens[captionIndex]);
      captionIndex += 1;
    }
    if (!matched) continue;
    const windowLength = ranges.at(-1).end - ranges[0].start;
    const maximumWindow = Math.max(96, String(unit || "").length * 5);
    if (windowLength > maximumWindow) continue;
    if (!best || windowLength < best.windowLength) best = {ranges, windowLength};
  }
  return best?.ranges || [];
}

function highlightSpan(caption, span, unit) {
  const text = String(caption || "");
  if (
    span &&
    typeof span === "object" &&
    Number.isInteger(span.start) &&
    Number.isInteger(span.end) &&
    span.start >= 0 &&
    span.end >= span.start &&
    span.end <= text.length
  ) {
    return {html: renderHighlightRanges(text, [span]), mode: "extractor"};
  }
  const needle = String(span || "");
  if (needle) {
    const index = text.toLocaleLowerCase().indexOf(needle.toLocaleLowerCase());
    if (index >= 0) {
      return {
        html: renderHighlightRanges(text, [{start: index, end: index + needle.length}]),
        mode: "extractor",
      };
    }
  }
  const lexicalRanges = lexicalCueRanges(text, unit);
  if (lexicalRanges.length) {
    return {html: renderHighlightRanges(text, lexicalRanges, "lexical-cue"), mode: "lexical"};
  }
  return {html: escapeHtml(text), mode: "none"};
}

function choice(name, value, title, detail) {
  const selected = state.answers[name] === value ? " selected" : "";
  return `<button class="choice${selected}" type="button" aria-pressed="${selected ? "true" : "false"}" data-answer="${escapeHtml(name)}" data-value="${escapeHtml(value)}">
    <span class="choice-copy"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></span>
  </button>`;
}

function studyExitControls() {
  if (state.practiceMode || state.testMode) {
    return `<button class="withdraw-button" id="withdraw-study" type="button">${state.practiceMode ? "Stop rehearsal" : "Stop test"}</button>`;
  }
  return `<button class="pause-button" id="pause-study" type="button">Pause &amp; leave</button>
    <details class="study-options">
      <summary>More</summary>
      <div class="study-options-menu">
        <strong>Participation options</strong>
        <span>Pausing keeps every saved response and lets you return later.</span>
        <button class="withdraw-button" id="withdraw-study" type="button">Withdraw permanently…</button>
      </div>
    </details>`;
}

function bindStudyExitControls() {
  app.querySelector("#pause-study")?.addEventListener("click", pauseStudy);
  app.querySelector("#withdraw-study")?.addEventListener("click", withdrawStudy);
}

function header(task, progress, {readOnly = false} = {}) {
  const phase = task.phase;
  const completed = Number(progress?.completed || 0);
  const total = Math.max(1, Number(progress?.total || 1));
  const percentage = Math.min(100, Math.round((completed / total) * 100));
  return `<header class="topbar">
    <div class="brand">
      <span class="brand-mark">C</span>
      <span class="brand-copy"><strong>Visual Claim Study</strong><span>${state.practiceMode ? "Author rehearsal · answers discarded" : state.testMode ? "Participant test · no research data retained" : "Human anchor · blinded"}</span></span>
    </div>
    <nav class="phase-track" aria-label="Study phases">
      <span class="phase-node ${phase === "caption" ? "active" : "done"}">
        <span class="phase-number">${phase === "image" ? "✓" : "1"}</span><span>Caption claim</span>
      </span>
      <span class="phase-line"></span>
      <span class="phase-node ${phase === "image" ? "active" : ""}">
        <span class="phase-number">2</span><span>Image evidence</span>
      </span>
    </nav>
    <div class="study-actions">
      <div class="progress-summary">
        <div class="progress-line"><strong>${completed + 1} / ${total}</strong><span>${percentage}% complete</span></div>
        <div class="progress-bar" role="progressbar" aria-label="Current phase progress" aria-valuemin="0" aria-valuemax="${total}" aria-valuenow="${completed}" style="--progress:${percentage}%"><span></span></div>
      </div>
      ${state.previousTask && !readOnly ? `<button class="previous-button" id="previous-screen" type="button" title="View the previous screen read only">← Back</button>` : ""}
      ${!readOnly ? `<button class="guide-button" id="open-guide" type="button">Guide</button>` : ""}
      ${studyExitControls()}
    </div>
  </header>`;
}

function mobileClaimSummary(task) {
  return `<div class="mobile-claim-summary">
    <span class="eyebrow">${task.phase === "caption" ? "Current claim · highlights only help locate caption words" : state.imagePrimaryLocked ? "Claim-level support locked · whole-pair comparison below" : "Same claim · now judge against the image"}</span>
    <strong>${escapeHtml(task.unit)}</strong>
    <span>${escapeHtml(task.category.replaceAll("_", " "))} · target: ${escapeHtml(task.target || "scene")}</span>
  </div>`;
}

function captionEvidence(task) {
  const highlighted = highlightSpan(task.caption, task.span, task.unit);
  const highlightHelp =
    highlighted.mode === "extractor"
      ? "The yellow highlight shows the extractor-provided evidence span."
      : highlighted.mode === "lexical"
        ? "The linked blue highlights show exact claim words found in order within one short stretch of the caption. Words between them remain part of the context; this cue is not proof that the full claim is stated."
        : "No reliable evidence span or short lexical match was available. Read the caption and decide whether it states the complete claim.";
  return `<section class="panel evidence-panel">
    <div class="panel-heading">
      <div class="heading-label"><span class="eyebrow">Caption evidence</span><span class="privacy-badge">${icons.lock} Image hidden</span></div>
      <span class="budget-badge">Fixed audit window</span>
    </div>
    ${mobileClaimSummary(task)}
    <div class="mobile-caption-task-cue">
      <span>Your answer area</span>
      <strong>Three questions are waiting below the caption.</strong>
      <button type="button" id="mobile-go-questions">Go to answer choices ↓</button>
    </div>
    <div class="caption-stage">
      <div class="stage-intro">
        <div class="phase-purpose">
          <strong>Step 1 uses words only—not the image.</strong>
          <span>Read this fixed caption window as evidence, then answer the highlighted questions on the right. The paired image appears later.</span>
        </div>
        <h1>Caption evidence <span aria-hidden="true">→</span></h1>
      </div>
      <article class="caption-paper">
        ${
          highlighted.mode === "lexical"
            ? `<span class="caption-cue-label"><b>↔</b> Linked lexical cue · read blue words together</span>`
            : ""
        }
        <p class="caption-text">${highlighted.html}</p>
      </article>
      <p class="caption-footnote">${icons.info}<span>${highlightHelp} The caption text above is the complete fixed window you are asked to judge. The highlight is only a pointer; use the questions on the right to give your answer.</span></p>
    </div>
  </section>`;
}

function imageEvidence(task, {forceRevealed = false} = {}) {
  const revealed = forceRevealed || state.imageRevealed;
  const coverageCaption =
    state.imagePrimaryLocked && typeof state.coverageCaption === "string"
      ? state.coverageCaption
      : null;
  return `<section class="panel evidence-panel">
    <div class="panel-heading">
      <div class="heading-label"><span class="eyebrow">Image evidence</span><span class="privacy-badge">${icons.lock} ${coverageCaption ? "Caption revealed after lock" : "Caption hidden"}</span></div>
      <span class="budget-badge">${coverageCaption ? "Whole-pair comparison" : "Visible evidence only"}</span>
    </div>
    ${mobileClaimSummary(task)}
    ${state.pairTransition ? `<div class="pair-transition"><span>✓ Caption judgment saved and locked</span><strong>Now check the same yellow claim against the image ↓</strong></div>` : ""}
    <div class="image-stage ${coverageCaption ? "coverage-stage" : ""}">
      <div class="image-frame ${revealed ? "" : "image-covered"}" id="image-frame">
        ${
          revealed
            ? `<img data-protected-image alt="Study image for the current visual claim" />
              <div class="image-toolbar">
                <button class="icon-button" id="zoom-image" type="button" aria-label="Open larger image view">${icons.zoom}</button>
              </div>`
            : `<div class="image-safety-curtain" role="region" aria-labelledby="image-safety-title">
                <span class="eyebrow">Image covered · applies to every task</span>
                <h1 id="image-safety-title" tabindex="-1">Reveal only when you are ready.</h1>
                <p>Public web images may contain unexpected or sensitive material, including partial nudity in non-sexual or cultural contexts, violence, or medical imagery. You may skip any image. This notice does not describe or classify this particular image.</p>
                <div class="image-safety-actions">
                  <button class="primary-button" id="reveal-image" type="button">Reveal image ${icons.arrow}</button>
                  <button class="secondary-button" id="skip-image-unseen" type="button">Skip without viewing</button>
                </div>
                <small>Skipping records “Prefer not to answer” and shows the next claim.</small>
              </div>`
        }
      </div>
      ${
        coverageCaption
          ? `<article class="coverage-caption" aria-labelledby="coverage-caption-title">
              <span class="eyebrow">Paired caption · revealed after support was locked</span>
              <h2 id="coverage-caption-title">Now compare the displayed caption window with the image.</h2>
              <p>${escapeHtml(coverageCaption)}</p>
            </article>`
          : ""
      }
    </div>
    ${
      revealed
        ? `<dialog class="image-dialog" id="image-dialog">
            <div class="dialog-toolbar"><span>Full-resolution inspection</span><button type="button" id="close-image-dialog">Close</button></div>
            <div class="dialog-canvas"><img data-protected-image alt="Full-resolution study image" /></div>
          </dialog>`
        : ""
    }
  </section>`;
}

const categoryDefinitions = {
  object: "a named visible entity or scene element",
  attribute: "a visible property attached to a target",
  relation: "a visible spatial, action, or participant relationship",
  count: "an exact visible number of a target",
  style: "a visible aesthetic or rendering appearance",
  camera: "visible viewpoint, framing, focus, or composition",
  lighting: "visible illumination, shadow, or light quality",
  text_rendering: "a specific string rendered legibly in the image",
};

function claimLengthClass(unit) {
  const length = String(unit || "").length;
  if (length > 42) return "claim-title-long";
  if (length > 28) return "claim-title-medium";
  return "claim-title-short";
}

function captionQuestionMap() {
  return `<section class="question-map" aria-label="Three main caption questions">
    <div class="question-map-heading">
      <strong id="question-map-status">Question 1 / 3</strong>
      <span>3 main questions</span>
    </div>
    <div class="question-map-track" aria-label="Jump to a caption question">
      <button type="button" data-question-step="1" aria-label="Go to Question 1">1</button>
      <i></i>
      <button type="button" data-question-step="2" aria-label="Go to Question 2">2</button>
      <i></i>
      <button type="button" data-question-step="3" aria-label="Go to Question 3">3</button>
    </div>
    <p>Highlighted evidence may be one continuous span or nearby linked words. Read linked words together, then judge the whole claim. If Question 3 is “No,” one required corrected-type follow-up appears.</p>
  </section>`;
}

function updateCaptionQuestionMap() {
  if (state.task?.phase !== "caption") return;
  const done = [
    Boolean(state.answers.caption_licensed),
    Boolean(state.answers.atomic_visual_claim),
    Boolean(state.answers.category_check) &&
      (state.answers.category_check !== "incorrect" || Boolean(state.answers.corrected_category)),
  ];
  const allDone = done.every(Boolean);
  const current = allDone ? 3 : done.findIndex((value) => !value) + 1;
  const status = app.querySelector("#question-map-status");
  if (status) status.textContent = allDone ? "3 / 3 complete" : `Question ${current} / 3`;
  app.querySelectorAll("[data-question-step]").forEach((step) => {
    const index = Number(step.dataset.questionStep) - 1;
    step.classList.toggle("complete", done[index]);
    step.classList.toggle("active", !allDone && index === current - 1);
  });
}

function setCaptionQuestionView(stepNumber) {
  const status = app.querySelector("#question-map-status");
  if (status) status.textContent = `Question ${stepNumber} / 3`;
  app.querySelectorAll("[data-question-step]").forEach((step) => {
    step.classList.toggle("active", Number(step.dataset.questionStep) === stepNumber);
  });
}

function captionQuestions(task) {
  const categorySkipped = state.answers.category_check === "not_applicable";
  return `<div class="question-block question-start">
    <div class="attention-cue" role="note"><strong>Start here</strong><span>Answer using the caption on the left. After each answer, the next question moves into view.</span></div>
    <span class="question-number">Question 1 · Caption wording</span>
    <h2 id="caption-license-question" tabindex="-1">Does the caption say the whole meaning of <span class="inline-claim-ref">this claim</span>?</h2>
    <div class="choice-list choice-grid-3" role="group" aria-labelledby="caption-license-question">
      ${choice("caption_licensed", "yes", "Yes", "The caption says the complete claim.")}
      ${choice("caption_licensed", "no", "No", "The caption leaves out some or all of it.")}
      ${choice("caption_licensed", "uncertain", "Not sure", "The wording is too ambiguous to decide.")}
    </div>
  </div>
  <div class="question-block">
    <span class="question-number">Question 2 · One visual detail</span>
    <div class="question-title-row">
      <h2 id="atomic-claim-question" tabindex="-1">Is this one independently checkable visual detail?</h2>
      <details class="rulebook">
        <summary aria-label="Open or close the one-detail guide" title="One-detail guide">?</summary>
        <div class="rulebook-card" role="note">
          <div class="rulebook-card-heading"><strong>One-detail guide</strong><button type="button" data-close-rulebook aria-label="Close one-detail guide">×</button></div>
          <p><b>Yes · one detail:</b> one object, or one property, count, or configuration about one target—or one relationship. It may use several words.</p>
          <p><b>Not visual:</b> metadata, subjective filler, or wording that an image cannot settle.</p>
          <p><b>Several details:</b> independently checkable facts have been joined together.</p>
          <p><b>Not sure:</b> use this only when the boundary remains genuinely ambiguous.</p>
          <span>Quick test: could part of the phrase be removed and still leave a separate fact that could be true or false on its own? If yes, choose “several details.”</span>
        </div>
      </details>
    </div>
    <p class="micro-guidance"><strong>One detail can use several words.</strong> A named object such as “phoenix,” “large herd,” or “crowded ensemble” can itself be one detail. So can one property or count about one target, or one relationship. “Pulled back in a low ponytail” stays together as one hairstyle detail. Choose “several details” for facts that could vary independently, such as “wearing a red shirt and holding a cup.”</p>
    <div class="choice-list choice-grid-2" role="group" aria-labelledby="atomic-claim-question">
      ${choice("atomic_visual_claim", "yes", "Yes · one detail", "One object, one property/count/configuration, or one relationship.")}
      ${choice("atomic_visual_claim", "not_visual", "No · not visual", "An image cannot settle this claim.")}
      ${choice("atomic_visual_claim", "not_atomic", "No · several details", "Independent visual facts are joined together.")}
      ${choice("atomic_visual_claim", "uncertain", "Not sure", "I cannot confidently decide.")}
    </div>
  </div>
  <div class="question-block" id="category-question">
    <span class="question-number">Question 3 · Visual type</span>
    <h2 id="category-check-question" tabindex="-1">Does the type <span class="type-ref">${escapeHtml(task.category.replaceAll("_", " "))}</span> fit this claim?</h2>
    <p class="micro-guidance">This question checks whether the proposed visual type is the best fit for the claim.</p>
    <div class="category-active ${categorySkipped ? "hidden" : ""}">
      <p class="proposed-type">
        <span>Proposed visual type</span>
        <strong>${escapeHtml(task.category.replaceAll("_", " "))}</strong>
        <span>${escapeHtml(categoryDefinitions[task.category] || "")}</span>
      </p>
      <div class="choice-list choice-grid-3" role="group" aria-labelledby="category-check-question">
        ${choice("category_check", "correct", "Yes", "This visual type is the best fit.")}
        ${choice("category_check", "incorrect", "No", "Another visual type fits better.")}
        ${choice("category_check", "uncertain", "Not sure", "I cannot confidently choose a type.")}
      </div>
      <label class="correction-field ${state.answers.category_check === "incorrect" ? "" : "hidden"}" id="correction-field">
        <span class="eyebrow">Required follow-up · corrected type</span>
        <span>Choose the visual type that fits better before saving.</span>
        <select id="corrected-category">
          <option value="">Choose one…</option>
          ${["object", "attribute", "relation", "count", "style", "camera", "lighting", "text_rendering"]
            .filter((value) => value !== task.category)
            .map((value) => `<option value="${value}" ${state.answers.corrected_category === value ? "selected" : ""}>${value.replaceAll("_", " ")}</option>`)
            .join("")}
        </select>
      </label>
    </div>
    <p class="category-skipped ${categorySkipped ? "" : "hidden"}" id="category-skipped">Not applicable because Question 2 marks this as non-visual or as several separate details.</p>
  </div>`;
}

const categoryGuidance = {
  count: "<strong>Count:</strong> the exact number and target must be visible; occluded or unbounded sets are uncertain.",
  relation: "<strong>Relation:</strong> both participants and the stated relation must be visibly recoverable.",
  text_rendering: "<strong>Rendered text:</strong> the required string must be legible; unreadable text is uncertain.",
  attribute: "<strong>Attribute:</strong> verify that the attribute belongs to the stated target, not merely the scene.",
  camera: "<strong>Camera:</strong> judge visible composition only, not inferred equipment or production intent.",
  style: "<strong>Style:</strong> judge visible appearance, not an assumed artist, tool, or production process.",
  lighting: "<strong>Lighting:</strong> judge visible illumination and shadows, not an inferred light source outside the frame.",
  object: "<strong>Object:</strong> the named object must be visibly present at a judgeable scale.",
};

const imageReasonTags = [
  ["absent", "Absent"],
  ["contradicted", "Contradicted"],
  ["wrong_target", "Wrong target"],
  ["exact_count_mismatch", "Count mismatch"],
  ["text_mismatch", "Text mismatch"],
  ["occluded", "Occluded"],
  ["too_small", "Too small"],
  ["unreadable", "Unreadable"],
  ["ambiguous_referent", "Ambiguous referent"],
];

function reasonButton(value, title) {
  const selected = (state.answers.reason_tags || []).includes(value);
  return `<button type="button" class="reason-button ${selected ? "selected" : ""}" aria-pressed="${selected}" data-reason="${value}">${title}</button>`;
}

function imageQuestions(task) {
  return `<div class="question-block">
    <span class="question-number">Question 1 · Visual support</span>
    <h2 id="image-support-question">Does the image visibly support the complete claim?</h2>
    <div class="choice-list" role="group" aria-labelledby="image-support-question">
      ${choice("image_support", "yes", "Supported", "Visible evidence supports the complete claim.")}
      ${choice("image_support", "no", "Unsupported", "The claim is contradicted or lacks visible support.")}
      ${choice("image_support", "uncertain", "Uncertain", "Occlusion, scale, ambiguity, or unreadability prevents a decision.")}
      ${choice("image_support", "not_visual", "Not a visual claim", "The candidate cannot be decided from visual evidence.")}
      ${choice("image_support", "image_unavailable", "Image unusable", "The image is missing, corrupt, or cannot be inspected.")}
      ${choice("image_support", "prefer_not_to_answer", "Prefer not to answer", "Skip this image without giving a support label.")}
    </div>
    <p class="micro-guidance">${categoryGuidance[task.category] || categoryGuidance.object}</p>
  </div>
  <div class="question-block">
    <span class="question-number">Optional · issue</span>
    <h2 id="reason-question">Did anything make the claim unsupported or hard to judge?</h2>
    <p class="micro-guidance">Leave this blank when no issue applies. Select only the issue(s) that affected your judgment.</p>
    <div class="reason-grid" role="group" aria-labelledby="reason-question">
      ${imageReasonTags.map(([value, title]) => reasonButton(value, title)).join("")}
    </div>
  </div>
  <div class="question-block">
    <span class="question-number">Optional · confidence</span>
    <h2 id="confidence-question">How confident are you in this judgment?</h2>
    <div class="confidence-row" role="group" aria-labelledby="confidence-question">
      ${[[1, "Low"], [3, "Medium"], [5, "High"]].map(([value, label]) => `<button type="button" class="confidence-button ${state.answers.confidence === value ? "selected" : ""}" aria-pressed="${state.answers.confidence === value}" data-answer="confidence" data-value="${value}">${label}</button>`).join("")}
    </div>
  </div>`;
}

function imageUsefulnessQuestion() {
  const usefulness =
    state.answers.image_support === "yes"
      ? `<div class="question-block exploratory-block" role="region" aria-labelledby="usefulness-question">
    <span class="question-number">Exploratory · control usefulness</span>
    <h2 id="usefulness-question" tabindex="-1">How useful would this supported detail be for controlling what a text-to-image model should draw?</h2>
    <div class="utility-scale" role="group" aria-labelledby="usefulness-question">
      ${[1, 2, 3, 4, 5]
        .map(
          (value) =>
            `<button type="button" class="utility-button ${String(state.answers.control_usefulness) === String(value) ? "selected" : ""}" aria-pressed="${String(state.answers.control_usefulness) === String(value)}" data-answer="control_usefulness" data-value="${value}"><strong>${value}</strong><span>${value === 1 ? "Not useful" : value === 3 ? "Moderately useful" : value === 5 ? "Highly useful" : ""}</span></button>`,
        )
        .join("")}
      <button type="button" class="utility-button utility-na ${state.answers.control_usefulness === "cannot_judge" ? "selected" : ""}" aria-pressed="${state.answers.control_usefulness === "cannot_judge"}" data-answer="control_usefulness" data-value="cannot_judge"><strong>—</strong><span>Cannot judge</span></button>
    </div>
    <p class="exploratory-note">Optional. You may save without choosing a score. It is reported separately from the one-detail and image-support judgments.</p>
  </div>`
      : "";
  return `<div class="locked-summary" role="status">
      <span>Claim-level image support is locked</span>
      <strong>${escapeHtml(
        {
          yes: "Supported",
          no: "Unsupported",
          uncertain: "Uncertain",
          not_visual: "Not a visual claim",
        }[state.answers.image_support] || state.answers.image_support,
      )}</strong>
      <small>The paired caption was revealed only after this answer was locked.</small>
    </div>
    <div class="question-block coverage-block" role="region" aria-labelledby="coverage-question">
      <span class="question-number">Required · displayed-window coverage</span>
      <h2 id="coverage-question" tabindex="-1">How well does the displayed caption window cover the important visible content in the image?</h2>
      <p class="micro-guidance">Judge exactly the fixed caption window shown against the whole image. It may end mid-sentence; do not infer omitted continuation. This is a human salient-content rating, not an exhaustive count of every possible fact.</p>
      <div class="coverage-scale" role="group" aria-labelledby="coverage-question">
        ${[
          [1, "Very little"],
          [2, "Limited"],
          [3, "Moderate"],
          [4, "Good"],
          [5, "Nearly complete"],
        ]
          .map(
            ([value, label]) =>
              `<button type="button" class="coverage-button ${state.answers.salient_coverage === value ? "selected" : ""}" aria-pressed="${state.answers.salient_coverage === value}" data-answer="salient_coverage" data-value="${value}"><strong>${value}</strong><span>${label}</span></button>`,
          )
          .join("")}
      </div>
      <p class="exploratory-note">Required after a viewable image. Reported separately from the claim-level judgments.</p>
    </div>
    ${usefulness}`;
}

function imageRevealPending() {
  return `<div class="reveal-pending" role="status">
    <span class="question-number">Image decision</span>
    <h2>The image is still covered.</h2>
    <p>Use “Reveal image” to inspect it, or “Skip without viewing” to continue without a support label.</p>
  </div>`;
}

function responseComplete(task) {
  if (task.phase === "image") {
    if (!state.imagePrimaryLocked) return Boolean(state.answers.image_support);
    return Number.isInteger(state.answers.salient_coverage);
  }
  const corrected =
    state.answers.category_check !== "incorrect" ||
    (Boolean(state.answers.corrected_category) &&
      state.answers.corrected_category !== task.category);
  return Boolean(
    state.answers.caption_licensed &&
      state.answers.atomic_visual_claim &&
      state.answers.category_check &&
      corrected,
  );
}

function scrollToTaskElement(selector, {focus = true, block = "center"} = {}) {
  const target = app.querySelector(selector);
  if (!target) return;
  const reducedMotion =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  requestAnimationFrame(() => {
    target.scrollIntoView({behavior: reducedMotion ? "auto" : "smooth", block});
    if (focus) {
      target.focus({preventScroll: true});
    }
  });
}

function advanceCaptionAfterAnswer(answerName, answerValue, previousValue) {
  if (state.task?.phase !== "caption") return;
  if (previousValue !== undefined) return;
  if (answerName === "caption_licensed") {
    if (!state.answers.atomic_visual_claim) {
      scrollToTaskElement("#atomic-claim-question");
    }
    return;
  }
  if (answerName === "atomic_visual_claim") {
    if (["not_visual", "not_atomic"].includes(answerValue)) {
      if (responseComplete(state.task)) {
        scrollToTaskElement("#save-next", {focus: false, block: "nearest"});
      }
      return;
    }
    if (
      !state.answers.category_check ||
      state.answers.category_check === "not_applicable"
    ) {
      scrollToTaskElement("#category-check-question");
    } else if (responseComplete(state.task)) {
      scrollToTaskElement("#save-next", {focus: false, block: "nearest"});
    }
    return;
  }
  if (answerName === "category_check") {
    if (answerValue === "incorrect") {
      scrollToTaskElement("#corrected-category");
    } else if (responseComplete(state.task)) {
      scrollToTaskElement("#save-next", {focus: false, block: "nearest"});
    }
  }
}

async function loadProtectedImage(task) {
  if (!task.image_url) return;
  const headers = new Headers();
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  try {
    const response = await fetch(task.image_url, {headers, cache: "no-store"});
    if (!response.ok) throw new Error(`Image request failed (${response.status})`);
    const objectUrl = URL.createObjectURL(await response.blob());
    if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
    state.imageObjectUrl = objectUrl;
    app.querySelectorAll("[data-protected-image]").forEach((image) => {
      image.src = objectUrl;
    });
  } catch (error) {
    const warning = document.createElement("p");
    warning.className = "image-error";
    warning.setAttribute("role", "alert");
    warning.textContent = "This image could not be displayed. Choose “Image unusable.”";
    app.querySelector("#image-frame")?.append(warning);
  }
}

function renderPreviousScreen() {
  const previous = state.previousTask;
  if (!previous || !state.task) return;
  state.reviewingPrevious = true;
  if (state.imageObjectUrl) {
    URL.revokeObjectURL(state.imageObjectUrl);
    state.imageObjectUrl = null;
  }
  app.innerHTML = `${header(previous, state.bootstrap?.progress || {}, {readOnly: true})}
    <main class="study-grid previous-review-grid">
      ${previous.phase === "caption" ? captionEvidence(previous) : imageEvidence(previous, {forceRevealed: true})}
      <aside class="panel previous-review-panel">
        <span class="eyebrow">Previous screen · read only</span>
        <h1 class="claim-title ${claimLengthClass(previous.unit)}"><span>${escapeHtml(previous.unit)}</span></h1>
        <p>The saved response is locked. Viewing a prior screen cannot change the human-study record.</p>
        <button class="primary-button" id="return-current" type="button">Return to current claim ${icons.arrow}</button>
      </aside>
    </main>`;
  if (previous.phase === "image") loadProtectedImage(previous);
  app.querySelector("#return-current")?.addEventListener("click", () => {
    state.reviewingPrevious = false;
    renderTask(state.task, state.bootstrap?.progress || {});
  });
  bindStudyExitControls();
}

function tutorialSceneSvg() {
  return `<svg class="tutorial-scene" viewBox="0 0 560 320" role="img" aria-labelledby="tutorial-scene-title tutorial-scene-description">
    <title id="tutorial-scene-title">A red sphere above a green rectangular platform</title>
    <desc id="tutorial-scene-description">A simple synthetic illustration used only to explain the study interface.</desc>
    <defs>
      <radialGradient id="tutorial-sphere" cx="34%" cy="27%">
        <stop offset="0" stop-color="#ff9282"></stop><stop offset="0.48" stop-color="#df4f3e"></stop><stop offset="1" stop-color="#9f2f25"></stop>
      </radialGradient>
      <linearGradient id="tutorial-platform" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#65a983"></stop><stop offset="1" stop-color="#247050"></stop>
      </linearGradient>
      <filter id="tutorial-shadow"><feGaussianBlur stdDeviation="8"></feGaussianBlur></filter>
    </defs>
    <rect width="560" height="320" rx="28" fill="#f4f1e9"></rect>
    <ellipse cx="280" cy="252" rx="112" ry="18" fill="#2d3b36" opacity=".16" filter="url(#tutorial-shadow)"></ellipse>
    <rect x="125" y="222" width="310" height="54" rx="8" fill="url(#tutorial-platform)"></rect>
    <path d="M125 234h310" stroke="#9bd1ae" stroke-width="2" opacity=".7"></path>
    <ellipse cx="280" cy="203" rx="62" ry="11" fill="#27332f" opacity=".18"></ellipse>
    <circle cx="280" cy="126" r="64" fill="url(#tutorial-sphere)"></circle>
    <circle cx="257" cy="103" r="14" fill="#ffd1c9" opacity=".52"></circle>
  </svg>`;
}

function tutorialChoice(name, value, title, description) {
  const selected = state.tutorialAnswers[name] === value;
  return `<button type="button" class="tutorial-choice ${selected ? "selected" : ""}" aria-pressed="${selected}" data-tutorial-answer="${name}" data-value="${value}">
    <strong>${title}</strong><span>${description}</span>
  </button>`;
}

function tutorialDetailChoices(name) {
  return [
    tutorialChoice(
      name,
      "yes",
      "Yes · one detail",
      "One object, one property/count/configuration, or one relationship.",
    ),
    tutorialChoice(
      name,
      "not_visual",
      "No · not visual",
      "An image cannot settle this claim.",
    ),
    tutorialChoice(
      name,
      "not_atomic",
      "No · several details",
      "Independent visual facts are joined together.",
    ),
    tutorialChoice(
      name,
      "uncertain",
      "Not sure",
      "I cannot confidently decide.",
    ),
  ].join("");
}

function tutorialClaimAnchor(label = "Claim to check") {
  return `<div class="tutorial-practice-claim">
    <span>${label}</span>
    <div><em>Attribute</em><small>Target · sphere</small></div>
    <strong>red</strong>
  </div>`;
}

function tutorialIntroduction() {
  return `<div class="tutorial-introduction">
    <section class="tutorial-intro-copy">
      <span class="eyebrow">Before the hands-on practice</span>
      <h1>What are we asking you to judge?</h1>
      <p>Image captions can sound detailed while leaving things out or adding things that are not visible. This study checks, one small statement at a time, whether an automated audit agrees with human judgment.</p>
      <p>You will first judge <strong>what the displayed caption window actually says</strong>. The image appears only afterward, so it cannot influence your text-only answer. Finally, you will compare that fixed text window with the whole image.</p>
      <div class="tutorial-purpose-flow" aria-label="The three judgment stages">
        <div><b>1</b><span><strong>Caption</strong>What does the text say?</span></div>
        <div><b>2</b><span><strong>Image</strong>Is the claim visibly supported?</span></div>
        <div><b>3</b><span><strong>Whole pair</strong>Does the caption cover the important content?</span></div>
      </div>
    </section>
    <aside class="tutorial-glossary" aria-label="Short study glossary">
      <span class="tutorial-panel-label">PLAIN-LANGUAGE GLOSSARY</span>
      <dl>
        <div><dt>Caption</dt><dd>The full text paired with an image.</dd></div>
        <div><dt>Claim</dt><dd>The short statement you are checking right now—for example, “red.”</dd></div>
        <div><dt>Target</dt><dd>The thing or scene part the claim refers to—for example, “sphere.”</dd></div>
        <div><dt>Object</dt><dd>A visible thing, person, animal, place, or scene element.</dd></div>
        <div><dt>Attribute</dt><dd>A visible property of a target, such as red, wooden, or smiling.</dd></div>
        <div class="tutorial-cbu-term"><dt>CBU</dt><dd>The research name for a claim that can be checked as <strong>one visual detail</strong>. You do not need to memorize this abbreviation; the questions use plain language.</dd></div>
      </dl>
      <p>Other visual types include relation, count, style, camera, lighting, and rendered text. Each task shows a short definition when needed.</p>
    </aside>
  </div>`;
}

function tutorialCaptionPractice() {
  return `<div class="tutorial-practice-grid">
    <section class="tutorial-practice-evidence">
      <span class="tutorial-panel-label">CAPTION · USE ONLY THIS TEXT FOR NOW</span>
      <h1>A glossy <mark>red</mark> sphere floats above a green rectangular platform.</h1>
      <div class="tutorial-hidden-image">${icons.lock}<span><strong>Image hidden</strong>Do not guess what it might show yet.</span></div>
      <p>The highlight helps you find relevant words. It is a pointer, not an answer.</p>
    </section>
    <aside class="tutorial-practice-response">
      ${tutorialClaimAnchor()}
      <div class="attention-cue" role="note"><strong>Start here</strong><span>Answer using the caption on the left, then try three short boundary checks.</span></div>
      <div class="tutorial-mini-question">
        <span>1 · Caption wording</span>
        <h2 id="tutorial-caption-question">Does the caption say the whole claim?</h2>
        <div class="tutorial-choice-grid" role="group" aria-labelledby="tutorial-caption-question">
          ${tutorialChoice("caption", "yes", "Yes", "The caption says “red.”")}
          ${tutorialChoice("caption", "no", "No", "Some or all is missing.")}
          ${tutorialChoice("caption", "uncertain", "Not sure", "The wording is ambiguous.")}
        </div>
      </div>
      <div class="tutorial-mini-question">
        <span>2 · One visual detail</span>
        <h2 id="tutorial-detail-question">Is “red” one independently checkable visual detail?</h2>
        <p>One detail describes one target—or one relationship—and may use several words. “Pulled back in a low ponytail” is still one hairstyle detail.</p>
        <div class="tutorial-choice-grid tutorial-choice-grid-2" role="group" aria-labelledby="tutorial-detail-question">
          ${tutorialDetailChoices("detail")}
        </div>
      </div>
      <div class="tutorial-mini-question">
        <span>3 · Visual type</span>
        <h2 id="tutorial-type-question">Does the type <b class="type-ref">attribute</b> fit “red”?</h2>
        <p>An attribute is a visible property of a target—here, the sphere's color.</p>
        <div class="tutorial-choice-grid" role="group" aria-labelledby="tutorial-type-question">
          ${tutorialChoice("type", "yes", "Yes", "Attribute is the best fit.")}
          ${tutorialChoice("type", "no", "No", "Another type fits better.")}
          ${tutorialChoice("type", "uncertain", "Not sure", "The type is ambiguous.")}
        </div>
      </div>
      <div class="tutorial-boundary-practice">
        <span class="tutorial-panel-label">THREE QUICK BOUNDARY CHECKS</span>
        <div class="tutorial-mini-question">
          <span>Caption does not say it</span>
          <h2 id="tutorial-missing-question">Does the caption say “metallic”?</h2>
          <p>The caption says “glossy,” but that does not automatically mean metallic. Judge what is stated, not what seems plausible.</p>
          <div class="tutorial-choice-grid" role="group" aria-labelledby="tutorial-missing-question">
            ${tutorialChoice("missing", "yes", "Yes", "Metallic is directly stated.")}
            ${tutorialChoice("missing", "no", "No", "Metallic is not stated.")}
            ${tutorialChoice("missing", "uncertain", "Not sure", "The wording is ambiguous.")}
          </div>
        </div>
        <div class="tutorial-mini-question">
          <span>Several independent details</span>
          <h2 id="tutorial-bundle-question">Is “glossy and red” one detail or several?</h2>
          <p>Glossiness and color could change independently; removing either still leaves a separate visual fact.</p>
          <div class="tutorial-choice-grid tutorial-choice-grid-2" role="group" aria-labelledby="tutorial-bundle-question">
            ${tutorialDetailChoices("bundle")}
          </div>
        </div>
        <div class="tutorial-mini-question">
          <span>Legitimate uncertainty</span>
          <h2 id="tutorial-uncertain-question">What about “close to it” with no clear target?</h2>
          <p>Because “it” is unresolved, the intended unit cannot be classified confidently. Do not guess.</p>
          <div class="tutorial-choice-grid tutorial-choice-grid-2" role="group" aria-labelledby="tutorial-uncertain-question">
            ${tutorialDetailChoices("ambiguity")}
          </div>
        </div>
      </div>
    </aside>
  </div>`;
}

function tutorialImagePractice() {
  return `<div class="tutorial-practice-grid">
    <section class="tutorial-practice-evidence tutorial-image-evidence">
      <span class="tutorial-panel-label">IMAGE · THE CAPTION IS NOW HIDDEN</span>
      ${tutorialSceneSvg()}
      <p>In the real task, every image starts behind a safety curtain. After revealing it, judge only what is visibly supported.</p>
    </section>
    <aside class="tutorial-practice-response">
      ${tutorialClaimAnchor("Same claim · now check the image")}
      <div class="attention-cue" role="note"><strong>Image question</strong><span>Use only the synthetic image on the left.</span></div>
      <div class="tutorial-mini-question">
        <span>Visual support</span>
        <h2 id="tutorial-support-question">Does the image visibly support the whole claim “red”?</h2>
        <div class="tutorial-choice-grid tutorial-choice-grid-2" role="group" aria-labelledby="tutorial-support-question">
          ${tutorialChoice("support", "yes", "Supported", "The sphere is visibly red.")}
          ${tutorialChoice("support", "no", "Unsupported", "The image contradicts or lacks it.")}
          ${tutorialChoice("support", "uncertain", "Not sure", "Visibility prevents a decision.")}
          ${tutorialChoice("support", "not_visual", "Not visual", "An image cannot settle the claim.")}
        </div>
      </div>
    </aside>
  </div>`;
}

function tutorialCoveragePractice() {
  return `<div class="tutorial-practice-grid">
    <section class="tutorial-practice-evidence tutorial-pair-evidence">
      <span class="tutorial-panel-label">WHOLE PAIR · IMAGE + CAPTION</span>
      ${tutorialSceneSvg()}
      <article>
        <span>Caption revealed after claim support was locked</span>
        <p>A glossy red sphere floats above a green rectangular platform.</p>
      </article>
    </section>
    <aside class="tutorial-practice-response">
      <div class="tutorial-practice-claim tutorial-locked-answer">
        <span>Earlier answer locked</span><strong>“red” is supported</strong>
      </div>
      <div class="attention-cue" role="note"><strong>2 final questions · 1 required + 1 optional</strong><span>Now compare the displayed caption window with the whole image.</span></div>
      <div class="tutorial-mini-question">
        <span>Displayed-window coverage</span>
        <h2 id="tutorial-coverage-question">How well does the displayed caption window cover the important visible content?</h2>
        <p>Use exactly the text shown. A real fixed window may end mid-sentence. This is a judgment scale, not a quiz with one correct number.</p>
        <div class="tutorial-coverage-scale" role="group" aria-labelledby="tutorial-coverage-question">
          ${[
            [1, "Very little"],
            [2, "Limited"],
            [3, "Moderate"],
            [4, "Good"],
            [5, "Nearly complete"],
          ]
            .map(([value, label]) => tutorialChoice("coverage", String(value), String(value), label))
            .join("")}
        </div>
      </div>
      <div class="tutorial-mini-question tutorial-optional-question">
        <span>Optional · control usefulness</span>
        <h2 id="tutorial-usefulness-question">How useful would this supported detail be for controlling what a text-to-image model should draw?</h2>
        <p>This appears only after you mark a claim as supported. It is optional and reported separately.</p>
        <div class="tutorial-usefulness-scale" role="group" aria-labelledby="tutorial-usefulness-question">
          ${[
            ["1", "Not useful"],
            ["2", ""],
            ["3", "Moderately useful"],
            ["4", ""],
            ["5", "Highly useful"],
            ["cannot_judge", "Cannot judge"],
          ]
            .map(([value, label]) => tutorialChoice("usefulness", value, value === "cannot_judge" ? "—" : value, label))
            .join("")}
        </div>
      </div>
    </aside>
  </div>`;
}

function tutorialComplete() {
  return `<div class="tutorial-complete">
    <span class="tutorial-complete-mark">✓</span>
    <span class="eyebrow">Practice complete</span>
    <h1>You have tried the full workflow.</h1>
    <p><strong>Caption first:</strong> answer three text-only questions. <strong>Image second:</strong> check visible support. <strong>Whole pair last:</strong> rate important-content coverage.</p>
    <p>Your real answers will not have a known “correct” choice. <em>Not sure</em> is the right response when evidence is genuinely ambiguous. Coverage is a judgment scale with no single correct number; use its anchors consistently.</p>
  </div>`;
}

function tutorialStepMarkup(step) {
  if (step === -1) return tutorialIntroduction();
  if (step === 0) return tutorialCaptionPractice();
  if (step === 1) return tutorialImagePractice();
  if (step === 2) return tutorialCoveragePractice();
  return tutorialComplete();
}

function tutorialActionReady(step) {
  if (step === -1) return true;
  if (step === 0) {
    return Boolean(
      state.tutorialAnswers.caption &&
      state.tutorialAnswers.detail &&
      state.tutorialAnswers.type &&
      state.tutorialAnswers.missing &&
      state.tutorialAnswers.bundle &&
      state.tutorialAnswers.ambiguity
    );
  }
  if (step === 1) return Boolean(state.tutorialAnswers.support);
  if (step === 2) return Boolean(state.tutorialAnswers.coverage);
  return true;
}

function showTutorialFeedback(message) {
  state.tutorialFeedback = message;
  const feedback = app.querySelector("#tutorial-feedback");
  if (feedback) {
    feedback.textContent = message;
    feedback.classList.toggle("success", Boolean(message) && !message.startsWith("Try again"));
    feedback.classList.toggle("hidden", !message);
  }
  if (tutorialAnnouncer) {
    tutorialAnnouncer.textContent = "";
    requestAnimationFrame(() => {
      tutorialAnnouncer.textContent = message;
    });
  }
}

function checkTutorialStep() {
  const step = state.tutorialStep;
  const correct =
    step === 0
      ? state.tutorialAnswers.caption === "yes" &&
        state.tutorialAnswers.detail === "yes" &&
        state.tutorialAnswers.type === "yes" &&
        state.tutorialAnswers.missing === "no" &&
        state.tutorialAnswers.bundle === "not_atomic" &&
        state.tutorialAnswers.ambiguity === "uncertain"
      : step === 1
        ? state.tutorialAnswers.support === "yes"
        : Boolean(state.tutorialAnswers.coverage);
  if (!correct) {
    state.tutorialFeedback =
      step === 0
        ? "Try again. “Red” is stated, is one detail, and is an attribute. “Metallic” is not stated. “Glossy and red” joins independent details. An unresolved target calls for “Not sure.”"
        : step === 1
          ? "Try again. The sphere is clearly visible and red, so the image supports the claim."
          : "";
    showTutorialFeedback(state.tutorialFeedback);
    return;
  }
  state.tutorialStep += 1;
  state.tutorialAnswers = {};
  state.tutorialFeedback =
    step === 0
      ? "Correct. The caption answer is now locked; next, use only the image."
      : step === 1
        ? "Correct. Image support is now locked; next, compare the displayed caption window and image."
        : "Coverage has no single correct number. Use the same anchors consistently across real items.";
  renderTutorial(state.tutorialTask, state.tutorialProgress);
}

function finishTutorial() {
  if (state.guideStartedAt !== null) {
    state.guideElapsedMs += performance.now() - state.guideStartedAt;
    state.guideStartedAt = null;
  }
  sessionStorage.setItem(tutorialStorageKey, "complete");
  state.tutorialStep = -1;
  state.tutorialAnswers = {};
  state.tutorialFeedback = "";
  state.tutorialReturnStep = null;
  const nextTask = state.tutorialTask;
  const nextProgress = state.tutorialProgress;
  state.tutorialTask = null;
  state.tutorialProgress = null;
  renderTask(nextTask, nextProgress);
}

function renderTutorial(task, progress, {restart = false} = {}) {
  if (restart) {
    state.tutorialStep = -1;
    state.tutorialAnswers = {};
    state.tutorialFeedback = "";
    state.tutorialReturnStep = null;
  }
  state.tutorialTask = task;
  state.tutorialProgress = progress;
  const step = Math.max(-1, Math.min(3, state.tutorialStep));
  const progressIndex = step + 1;
  const returning = Boolean(state.task);
  const labels = ["Caption practice", "Image practice", "Whole-pair practice"];
  const actionLabel =
    step === -1
      ? state.tutorialReturnStep === null
        ? "Got it · start the practice"
        : `Back to ${labels[state.tutorialReturnStep]}`
      : step === 0
      ? "Check caption answers"
      : step === 1
        ? "Check image answer"
        : step === 2
          ? "Check coverage answer"
          : returning
            ? "Return to current claim"
            : "Begin first real claim";
  app.innerHTML = `<main class="tutorial-layout">
    <div class="tutorial-status-banner" role="status">
      <strong>YOU ARE IN THE TUTORIAL</strong>
      <span>This is a synthetic practice example. Nothing you choose here is recorded.</span>
    </div>
    <header class="tutorial-header">
      <div class="brand"><span class="brand-mark">T</span><span class="brand-copy"><strong>Study guide &amp; practice</strong><span>Read the key terms, then try one synthetic example</span></span></div>
      <div class="tutorial-progress" aria-label="${step === 3 ? "Guide complete" : `Guide step ${progressIndex + 1} of 4`}">
        ${[0, 1, 2, 3].map((index) => `<span class="${index === progressIndex ? "active" : index < progressIndex || step === 3 ? "done" : ""}">${index + 1}</span>`).join("")}
      </div>
      ${studyExitControls()}
    </header>
    <section class="tutorial-card">
      <div class="tutorial-stage tutorial-practice-stage">${tutorialStepMarkup(step)}</div>
      <footer class="tutorial-footer">
        <span>${step === -1 ? "Purpose & key terms" : step < 3 ? labels[step] : "Ready for the study"}</span>
        <div class="tutorial-footer-action">
          ${step >= 0 && step < 3 ? `<button class="secondary-button" id="tutorial-back-to-terms" type="button">← Key terms</button>` : ""}
          ${returning && step !== 3 ? `<button class="secondary-button" id="tutorial-return-current" type="button">Return to current claim</button>` : ""}
          <p class="tutorial-feedback ${state.tutorialFeedback && !state.tutorialFeedback.startsWith("Try again") ? "success" : ""} ${state.tutorialFeedback ? "" : "hidden"}" id="tutorial-feedback">${escapeHtml(state.tutorialFeedback)}</p>
          <button class="primary-button" id="tutorial-next" type="button" ${tutorialActionReady(step) ? "" : "disabled"}>
            ${actionLabel} ${icons.arrow}
          </button>
        </div>
      </footer>
    </section>
  </main>`;
  app.querySelectorAll("[data-tutorial-answer]").forEach((button) => {
    button.addEventListener("click", () => {
      const name = button.dataset.tutorialAnswer;
      const previousValue = state.tutorialAnswers[name];
      state.tutorialAnswers[name] = button.dataset.value;
      app.querySelectorAll(`[data-tutorial-answer="${CSS.escape(name)}"]`).forEach((peer) => {
        const selected = peer === button;
        peer.classList.toggle("selected", selected);
        peer.setAttribute("aria-pressed", String(selected));
      });
      showTutorialFeedback("");
      const nextButton = app.querySelector("#tutorial-next");
      if (nextButton) nextButton.disabled = !tutorialActionReady(step);
      if (name === "coverage" && previousValue === undefined) {
        requestAnimationFrame(() => {
          app
            .querySelector(".tutorial-optional-question")
            ?.scrollIntoView({behavior: "auto", block: "center"});
        });
      }
    });
  });
  app.querySelector("#tutorial-next")?.addEventListener("click", () => {
    if (step === -1) {
      state.tutorialStep = state.tutorialReturnStep ?? 0;
      state.tutorialReturnStep = null;
      state.tutorialFeedback = "";
      renderTutorial(state.tutorialTask, state.tutorialProgress);
    } else if (step === 3) finishTutorial();
    else checkTutorialStep();
  });
  app.querySelector("#tutorial-back-to-terms")?.addEventListener("click", () => {
    state.tutorialReturnStep = step;
    state.tutorialStep = -1;
    state.tutorialFeedback = "";
    renderTutorial(state.tutorialTask, state.tutorialProgress);
  });
  app.querySelector("#tutorial-return-current")?.addEventListener("click", finishTutorial);
  bindStudyExitControls();
  showTutorialFeedback(state.tutorialFeedback);
  requestAnimationFrame(() => {
    window.scrollTo({top: 0, left: 0});
    const target = app.querySelector(
      ".tutorial-intro-copy h1, .attention-cue, .tutorial-complete h1",
    );
    target?.setAttribute("tabindex", "-1");
    target?.focus({preventScroll: true});
  });
}

function responsePanel(task) {
  const complete = responseComplete(task);
  const saveLabel =
    task.phase === "image" && !state.imageRevealed
      ? "Reveal or skip first"
      : task.phase === "image" &&
          !state.imagePrimaryLocked &&
          !["image_unavailable", "prefer_not_to_answer"].includes(state.answers.image_support)
        ? "Lock support & compare caption"
        : "Save & continue";
  return `<aside class="panel response-panel ${task.phase === "caption" ? "response-caption" : "response-image"}">
    <div class="claim-anchor">
      <div class="claim-meta">
        <span class="category-chip">${escapeHtml(task.category.replaceAll("_", " "))}</span>
        <span class="target-chip">Target · ${escapeHtml(task.target || "scene")}</span>
      </div>
      <span class="claim-title-label">${task.phase === "caption" ? "Claim to check · the highlight only helps locate caption wording" : state.imagePrimaryLocked ? "Same claim · image-support answer locked" : "Same claim · now judge against the image"}</span>
      <h1 class="claim-title ${claimLengthClass(task.unit)}"><span>${escapeHtml(task.unit)}</span></h1>
    </div>
    ${task.phase === "caption" ? captionQuestionMap() : ""}
    <div class="response-scroll">
      <p class="claim-hint">${task.phase === "caption" ? "For this step, ignore whether the image would make it true. Check only what the displayed caption window explicitly states." : state.imagePrimaryLocked ? "Support is locked. Now compare the displayed caption window with the entire image for the separate coverage rating." : "The caption is now hidden. Evaluate the same claim using only visible image evidence."}</p>
      ${task.phase === "caption"
        ? captionQuestions(task)
        : !state.imageRevealed
          ? imageRevealPending()
        : state.imagePrimaryLocked
          ? imageUsefulnessQuestion()
          : imageQuestions(task)}
    </div>
    <footer class="response-footer">
      <span class="blind-note">${icons.lock}<span>${task.phase === "caption" ? "3 main questions · “No” on visual type adds one required follow-up" : "Surface identity and automated judge outputs are hidden."}</span></span>
      <button class="primary-button" id="save-next" type="button" ${complete ? "" : "disabled"}>
        ${saveLabel} ${icons.arrow}
      </button>
    </footer>
  </aside>`;
}

function renderTask(task, progress) {
  const taskIdentity = task.assignment_id || task.item_id;
  const priorIdentity = state.task ? state.task.assignment_id || state.task.item_id : null;
  const priorKey = state.task ? `${priorIdentity}:${state.task.phase}` : null;
  const nextKey = `${taskIdentity}:${task.phase}`;
  const isNewTask = priorKey !== nextKey;
  if (isNewTask) {
    state.answers = {};
    state.captionCategoryDraft = null;
    state.imagePrimaryLocked = false;
    state.imageRevealed = false;
    state.coverageCaption = null;
    state.startedAt = performance.now();
    state.guideElapsedMs = 0;
    state.guideStartedAt = null;
    if (state.imageObjectUrl) {
      URL.revokeObjectURL(state.imageObjectUrl);
      state.imageObjectUrl = null;
    }
  }
  state.task = task;
  if (state.startedAt === null) state.startedAt = performance.now();
  app.innerHTML = `${header(task, progress)}
    <main class="study-grid">
      ${task.phase === "caption" ? captionEvidence(task) : imageEvidence(task)}
      ${responsePanel(task)}
    </main>`;

  app.querySelectorAll("[data-answer]").forEach((button) => {
    button.addEventListener("click", () => {
      const numericFields = new Set(["confidence", "control_usefulness", "salient_coverage"]);
      const rawValue = button.dataset.value;
      const previousValue = state.answers[button.dataset.answer];
      state.answers[button.dataset.answer] =
        numericFields.has(button.dataset.answer) && /^\d+$/.test(rawValue)
          ? Number(rawValue)
          : rawValue;
      app.querySelectorAll(`[data-answer="${CSS.escape(button.dataset.answer)}"]`).forEach((peer) => {
        peer.classList.toggle("selected", peer === button);
        peer.setAttribute("aria-pressed", String(peer === button));
      });
      if (button.dataset.answer === "category_check") {
        if (button.dataset.value !== "incorrect") delete state.answers.corrected_category;
        state.captionCategoryDraft = {
          category_check: button.dataset.value,
          corrected_category: state.answers.corrected_category,
        };
        app
          .querySelector("#correction-field")
          ?.classList.toggle("hidden", button.dataset.value !== "incorrect");
      }
      if (button.dataset.answer === "atomic_visual_claim") {
        const skipCategory = ["not_visual", "not_atomic"].includes(button.dataset.value);
        if (skipCategory) {
          if (
            state.answers.category_check &&
            state.answers.category_check !== "not_applicable"
          ) {
            state.captionCategoryDraft = {
              category_check: state.answers.category_check,
              corrected_category: state.answers.corrected_category,
            };
          }
          state.answers.category_check = "not_applicable";
          delete state.answers.corrected_category;
        } else if (state.answers.category_check === "not_applicable") {
          if (state.captionCategoryDraft?.category_check) {
            state.answers.category_check = state.captionCategoryDraft.category_check;
            if (state.captionCategoryDraft.corrected_category) {
              state.answers.corrected_category =
                state.captionCategoryDraft.corrected_category;
            }
          } else {
            delete state.answers.category_check;
          }
        }
        app.querySelector(".category-active")?.classList.toggle("hidden", skipCategory);
        app.querySelector("#category-skipped")?.classList.toggle("hidden", !skipCategory);
        app.querySelectorAll('[data-answer="category_check"]').forEach((peer) => {
          const selected = state.answers.category_check === peer.dataset.value;
          peer.classList.toggle("selected", selected);
          peer.setAttribute("aria-pressed", String(selected));
        });
        const correctionField = app.querySelector("#correction-field");
        correctionField?.classList.toggle(
          "hidden",
          state.answers.category_check !== "incorrect",
        );
        const correctedCategory = app.querySelector("#corrected-category");
        if (correctedCategory) {
          correctedCategory.value = state.answers.corrected_category || "";
        }
      }
      const saveButton = app.querySelector("#save-next");
      if (saveButton) saveButton.disabled = !responseComplete(state.task);
      updateCaptionQuestionMap();
      advanceCaptionAfterAnswer(
        button.dataset.answer,
        button.dataset.value,
        previousValue,
      );
    });
  });
  app.querySelectorAll("[data-reason]").forEach((button) => {
    button.addEventListener("click", () => {
      const reasons = new Set(state.answers.reason_tags || []);
      if (reasons.has(button.dataset.reason)) reasons.delete(button.dataset.reason);
      else reasons.add(button.dataset.reason);
      state.answers.reason_tags = [...reasons];
      button.classList.toggle("selected", reasons.has(button.dataset.reason));
      button.setAttribute("aria-pressed", String(reasons.has(button.dataset.reason)));
    });
  });
  const questionTargets = {
    1: "#caption-license-question",
    2: "#atomic-claim-question",
    3: "#category-check-question",
  };
  app.querySelectorAll("[data-question-step]").forEach((button) => {
    button.addEventListener("click", () => {
      const stepNumber = Number(button.dataset.questionStep);
      const target = app.querySelector(questionTargets[stepNumber]);
      setCaptionQuestionView(stepNumber);
      target?.scrollIntoView({behavior: "smooth", block: "center"});
      target?.focus({preventScroll: true});
    });
  });
  app.querySelector("[data-close-rulebook]")?.addEventListener("click", () => {
    app.querySelector(".rulebook")?.removeAttribute("open");
    app.querySelector(".rulebook summary")?.focus();
  });
  app.querySelector(".rulebook")?.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.currentTarget.removeAttribute("open");
    event.currentTarget.querySelector("summary")?.focus();
  });
  app.querySelector("#corrected-category")?.addEventListener("change", (event) => {
    state.answers.corrected_category = event.currentTarget.value || undefined;
    state.captionCategoryDraft = {
      category_check: state.answers.category_check,
      corrected_category: state.answers.corrected_category,
    };
    const saveButton = app.querySelector("#save-next");
    if (saveButton) saveButton.disabled = !responseComplete(state.task);
    updateCaptionQuestionMap();
    if (responseComplete(state.task)) {
      scrollToTaskElement("#save-next", {focus: false, block: "nearest"});
    }
  });
  const zoomButton = app.querySelector("#zoom-image");
  if (zoomButton) {
    zoomButton.addEventListener("click", () => {
      app.querySelector("#image-dialog")?.showModal();
    });
    app.querySelector("#close-image-dialog")?.addEventListener("click", () => {
      app.querySelector("#image-dialog")?.close();
    });
    app.querySelector("#image-dialog")?.addEventListener("click", (event) => {
      if (event.target === event.currentTarget) event.currentTarget.close();
    });
    app.querySelector("#image-frame img")?.addEventListener("error", () => {
      if (app.querySelector(".image-error")) return;
      const warning = document.createElement("p");
      warning.className = "image-error";
      warning.setAttribute("role", "alert");
      warning.textContent = "This image could not be displayed. Choose “Image unusable.”";
      app.querySelector("#image-frame")?.append(warning);
    });
    loadProtectedImage(task);
  }
  app.querySelector("#reveal-image")?.addEventListener("click", () => {
    state.imageRevealed = true;
    renderTask(state.task, state.bootstrap?.progress || {});
    requestAnimationFrame(() => {
      const question = app.querySelector("#image-support-question");
      question?.scrollIntoView({behavior: "smooth", block: "center"});
      question?.focus({preventScroll: true});
    });
  });
  app.querySelector("#skip-image-unseen")?.addEventListener("click", async () => {
    state.answers = {image_support: "prefer_not_to_answer"};
    await saveAndContinue();
  });
  app.querySelector("#save-next")?.addEventListener("click", saveAndContinue);
  bindStudyExitControls();
  app.querySelector("#previous-screen")?.addEventListener("click", renderPreviousScreen);
  app.querySelector("#open-guide")?.addEventListener("click", () => {
    state.guideStartedAt = performance.now();
    renderTutorial(state.task, state.bootstrap?.progress || {}, {restart: true});
  });
  app.querySelector("#mobile-go-questions")?.addEventListener("click", () => {
    const question = app.querySelector(".question-start");
    const reducedMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    question?.setAttribute("tabindex", "-1");
    question?.scrollIntoView({behavior: reducedMotion ? "auto" : "smooth", block: "start"});
    question?.focus({preventScroll: true});
  });
  updateCaptionQuestionMap();
  if (isNewTask) {
    requestAnimationFrame(() => {
      window.scrollTo({top: 0, left: 0});
      const scrollRegion = app.querySelector(".response-scroll");
      if (scrollRegion) scrollRegion.scrollTop = 0;
      const start = app.querySelector(
        task.phase === "caption" ? ".attention-cue" : ".claim-title",
      );
      start?.setAttribute("tabindex", "-1");
      start?.focus({preventScroll: true});
    });
  }
}

function renderLogin(error = "") {
  if (state.practiceMode) {
    app.innerHTML = `<main class="login-layout">
      <section class="login-card">
        <div class="brand"><span class="brand-mark">C</span><span class="brand-copy"><strong>Visual Claim Study</strong><span>Author rehearsal · answers discarded</span></span></div>
        <h1>One claim.<br />Two kinds of evidence.</h1>
        <p>Enter your private author-rehearsal code. Answer values are discarded; only volatile progress advances.</p>
        <form id="login-form">
          <label class="eyebrow" for="invite-code">Author rehearsal code</label>
          <input class="code-input" id="invite-code" name="code" autocomplete="one-time-code" spellcheck="false" required />
          ${error ? `<div class="error-banner" role="alert">${escapeHtml(error)}</div>` : ""}
          <div class="login-actions"><button class="primary-button" type="submit">Enter rehearsal ${icons.arrow}</button></div>
        </form>
      </section>
    </main>`;
    app.querySelector("#login-form").addEventListener("submit", login);
    return;
  }
  const savedCode = localStorage.getItem(participantCodeStorageKey);
  app.innerHTML = `<main class="login-layout">
    <section class="login-card enrollment-login-card">
      <div class="brand"><span class="brand-mark">C</span><span class="brand-copy"><strong>Visual Claim Study</strong><span>${state.testMode ? "Participant test · no research data retained" : "Human anchor · blinded"}</span></span></div>
      ${state.testMode ? `<div class="test-mode-banner" role="status"><strong>Response-discarding test mode</strong><span>This browser remembers only your temporary code. The test server keeps answers and progress in memory so you can test resume; all disappear when the server restarts and none are exported as research data.</span></div>` : ""}
      <h1>Check what captions say—and what images show.</h1>
      <p>You will judge one short visual claim at a time. First-time participants receive an anonymous resume code; returning participants continue with the same code. No name or email is requested.</p>
      ${error ? `<div class="error-banner" role="alert">${escapeHtml(error)}</div>` : ""}
      <div class="enrollment-actions">
        ${savedCode ? `<button class="primary-button" id="continue-saved-code" type="button">${state.testMode ? "Resume temporary test on this device" : "Continue on this device"} ${icons.arrow}</button>` : ""}
        <button class="${savedCode ? "secondary-button" : "primary-button"}" id="claim-participant-code" type="button">I’m new · get a participant code</button>
      </div>
      <details class="returning-code-panel" ${savedCode ? "" : "open"}>
        <summary>I already have a code</summary>
        <form id="login-form">
          <label class="eyebrow" for="invite-code">Participant code</label>
        <input class="code-input" id="invite-code" name="code" autocomplete="one-time-code" spellcheck="false" required />
          <div class="login-actions"><button class="secondary-button" type="submit">Continue with this code</button></div>
        </form>
      </details>
      <div class="privacy-list">
        <div><strong>Pseudonymous</strong>No names or email addresses.</div>
        <div><strong>Blinded</strong>No surface or model identity.</div>
        <div><strong>Resumable</strong>${state.testMode ? "Progress lasts only while this test server is running. Keep the test code to check resume behavior." : "Your responses are saved transactionally. Keep the participant code to return from another browser."}</div>
      </div>
    </section>
  </main>`;
  app.querySelector("#login-form").addEventListener("submit", login);
  app.querySelector("#claim-participant-code").addEventListener("click", claimParticipantCode);
  app.querySelector("#continue-saved-code")?.addEventListener("click", () => {
    loginWithCode(savedCode);
  });
}

async function login(event) {
  event.preventDefault();
  const code = new FormData(event.currentTarget).get("code");
  await loginWithCode(code);
}

async function loginWithCode(code) {
  try {
    const result = await api("/api/login", {method: "POST", body: JSON.stringify({code})});
    if (!state.practiceMode) localStorage.setItem(participantCodeStorageKey, String(code));
    state.token = result.token;
    sessionStorage.setItem("human-cbu-session", result.token);
    sessionStorage.removeItem(tutorialStorageKey);
    await bootstrap();
  } catch (error) {
    renderLogin(error.message);
  }
}

async function claimParticipantCode(event) {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Creating your anonymous code…";
  try {
    const result = await api("/api/enroll", {
      method: "POST",
      body: JSON.stringify({}),
    });
    if (typeof result.code !== "string" || typeof result.token !== "string") {
      throw new Error("The study could not issue a participant code.");
    }
    localStorage.setItem(participantCodeStorageKey, result.code);
    state.token = result.token;
    sessionStorage.setItem("human-cbu-session", result.token);
    sessionStorage.removeItem(tutorialStorageKey);
    renderIssuedParticipantCode(result.code);
  } catch (error) {
    renderLogin(error.message);
  }
}

function renderIssuedParticipantCode(code) {
  app.innerHTML = `<main class="login-layout">
    <section class="login-card issued-code-card">
      <div class="brand"><span class="brand-mark">✓</span><span class="brand-copy"><strong>Visual Claim Study</strong><span>${state.testMode ? "Temporary test code created" : "Anonymous code created"}</span></span></div>
      <span class="eyebrow">Save before continuing</span>
      <h1>This code is your way back.</h1>
      <p>It has been saved in this browser. Copy or download it as well if you may return from another browser or device. We cannot recover it from a name or email.</p>
      <div class="issued-code" role="status"><code id="issued-participant-code">${escapeHtml(code)}</code></div>
      <div class="issued-code-actions">
        <button class="secondary-button" id="copy-participant-code" type="button">Copy code</button>
        <button class="secondary-button" id="download-participant-code" type="button">Download .txt</button>
      </div>
      <label class="code-saved-check"><input id="code-saved-confirmation" type="checkbox" /><span>I saved my participant code somewhere I can find again.</span></label>
      <button class="primary-button" id="continue-after-code" type="button" disabled>Continue to study information ${icons.arrow}</button>
      <p class="storage-note">${state.testMode ? "This test code stops working when the response-discarding server restarts." : "On a shared device, remove the locally saved code after finishing or withdrawing."}</p>
    </section>
  </main>`;
  const confirmation = app.querySelector("#code-saved-confirmation");
  const continueButton = app.querySelector("#continue-after-code");
  confirmation.addEventListener("change", () => {
    continueButton.disabled = !confirmation.checked;
  });
  app.querySelector("#copy-participant-code").addEventListener("click", async (event) => {
    try {
      await navigator.clipboard.writeText(code);
      event.currentTarget.textContent = "Copied";
    } catch {
      window.getSelection()?.selectAllChildren(app.querySelector("#issued-participant-code"));
      event.currentTarget.textContent = "Select and copy the code";
    }
  });
  app.querySelector("#download-participant-code").addEventListener("click", () => {
    const objectUrl = URL.createObjectURL(
      new Blob([`Visual Claim Study participant code\n${code}\n`], {type: "text/plain"}),
    );
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = "visual-claim-study-participant-code.txt";
    link.click();
    URL.revokeObjectURL(objectUrl);
  });
  continueButton.addEventListener("click", bootstrap);
}

function participantInformationPanel(information) {
  if (!information) return "";
  return `<section class="participant-information" aria-labelledby="participant-information-title">
    <div class="participant-information-heading">
      <div>
        <span class="eyebrow">Participant notice</span>
        <h2 id="participant-information-title">What participation involves</h2>
      </div>
      <span class="approval-version">Version ${escapeHtml(information.approved_version)}</span>
    </div>
    <dl class="participant-information-grid">
      <div><dt>Expected duration</dt><dd>${escapeHtml(information.duration)}</dd></div>
      <div><dt>Compensation</dt><dd>${escapeHtml(information.compensation)}</dd></div>
      <div><dt>Data retention</dt><dd>${escapeHtml(information.data_retention)}</dd></div>
      <div><dt>Research contact</dt><dd>${escapeHtml(information.research_contact)}</dd></div>
      <div class="withdrawal-policy"><dt>Stopping or withdrawing</dt><dd>${escapeHtml(information.withdrawal_policy)}</dd></div>
    </dl>
    <a class="information-sheet-link" href="${escapeHtml(information.information_sheet_url)}" target="_blank" rel="noopener noreferrer">
      Open the full participant notice in a new tab ${icons.arrow}
    </a>
  </section>`;
}

function renderConsent(consentVersion, participantInformation) {
  app.innerHTML = `<main class="login-layout">
    <section class="login-card consent-card">
      <div class="brand"><span class="brand-mark">C</span><span class="brand-copy"><strong>Visual Claim Study</strong><span>Consent · ${escapeHtml(consentVersion || "")}</span></span></div>
      ${state.testMode ? `<div class="test-mode-banner" role="status"><strong>TEST MODE · NO RESEARCH DATA RETAINED</strong><span>Your code, choices, and progress are held temporarily so resume can be tested. They disappear when this server restarts and are never exported as research data.</span></div>` : ""}
      <h1>Before you begin.</h1>
      <div class="consent-copy">
        <p>You will judge short image–text claims for research on image–text dataset auditing. Public web images may contain unexpected or sensitive material, including partial nudity in non-sexual or cultural contexts, violence, or medical imagery. You may skip any image.</p>
        <p>Participation is voluntary. You may skip any image, stop now, or close the page at any time. ${state.testMode ? "This interface test temporarily holds choices only for testing progress and resume; it does not export them as research data." : "Responses are stored under a study-only pseudonym; the annotation database does not store names, email addresses, IP addresses, or browser identifiers."}</p>
      </div>
      ${participantInformationPanel(participantInformation)}
      <form id="consent-form">
        <label class="consent-check"><input type="checkbox" name="consented" required /><span>I understand the study and voluntarily agree to participate.</span></label>
        <input type="hidden" name="consent_version" value="${escapeHtml(consentVersion || "")}" />
        <div class="consent-actions">
          <button class="secondary-button" id="decline-consent" type="button">Decline</button>
          <button class="primary-button" type="submit">Agree &amp; begin ${icons.arrow}</button>
        </div>
      </form>
    </section>
  </main>`;
  app.querySelector("#consent-form").addEventListener("submit", submitConsent);
  app.querySelector("#decline-consent").addEventListener("click", () => submitConsent(null));
}

async function submitConsent(event) {
  if (event) event.preventDefault();
  const form = app.querySelector("#consent-form");
  const data = new FormData(form);
  const consented = Boolean(event);
  try {
    const result = await api("/api/consent", {
      method: "POST",
      body: JSON.stringify({
        consented,
        consent_version: data.get("consent_version"),
        profile: {},
      }),
    });
    if (!consented) {
      sessionStorage.removeItem("human-cbu-session");
      localStorage.removeItem("human-cbu-consent-version");
      localStorage.removeItem(participantCodeStorageKey);
      state.token = null;
      state.consentVersion = null;
      state.participantInformation = null;
      renderWaiting(
        "You declined participation. No annotation tasks were opened.",
        {title: "Participation declined.", allowWithdrawal: false},
      );
      return;
    }
    applyStudyResult(result);
  } catch (error) {
    window.alert(error.message);
  }
}

function recoverExpiredSession(message) {
  sessionStorage.removeItem("human-cbu-session");
  if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
  state.token = null;
  state.bootstrap = null;
  state.task = null;
  state.answers = {};
  state.startedAt = null;
  state.imagePrimaryLocked = false;
  state.imageRevealed = false;
  state.coverageCaption = null;
  state.imageObjectUrl = null;
  state.participantInformation = null;
  state.withdrawing = false;
  renderLogin(message);
}

async function recoverExpiredBootstrapSession() {
  sessionStorage.removeItem("human-cbu-session");
  state.token = null;
  let publicBootstrap = null;
  try {
    publicBootstrap = await api("/api/bootstrap");
  } catch {
    // Fail toward production wording; never promise response-discarding
    // behavior unless the current server states it explicitly.
  }
  state.practiceMode = Boolean(publicBootstrap?.practice_mode);
  state.testMode = Boolean(publicBootstrap?.test_mode);
  recoverExpiredSession(
    state.testMode
      ? "This temporary test session expired. Resume with the code saved on this device while the same test server is running. If the server restarted, choose “I’m new.”"
      : "Your session expired. Sign in again with your participant code to continue or withdraw.",
  );
}

async function resolveConsentVersion() {
  const remembered =
    state.consentVersion || localStorage.getItem("human-cbu-consent-version");
  if (remembered) return remembered;

  const result = await api("/api/bootstrap");
  const recovered =
    result.consent_version ||
    result.required_consent_version ||
    result.participant_information?.approved_version;
  if (typeof recovered !== "string" || !recovered) {
    throw new Error("The approved consent version could not be verified. Retry loading the study.");
  }
  state.consentVersion = recovered;
  localStorage.setItem("human-cbu-consent-version", recovered);
  return recovered;
}

async function withdrawStudy() {
  if (state.withdrawing) return;
  if (state.practiceMode || state.testMode) {
    const confirmed = window.confirm(
      state.practiceMode
        ? "Stop this author rehearsal? Its in-memory progress will be closed."
        : "Stop this response-discarding participant test?",
    );
    if (!confirmed) return;
  } else {
    const firstWarning = window.confirm(
      "Permanent withdrawal is different from pausing. It excludes every response saved under this participant code. Choose Cancel if you only want to stop for now.",
    );
    if (!firstWarning) return;
    const finalWarning = window.confirm(
      "Final warning: withdrawal cannot be undone, and this participant code cannot resume the study. Continue only if you want your saved responses excluded.",
    );
    if (!finalWarning) return;
  }
  const button = app.querySelector("#withdraw-study");
  const idleLabel = button?.textContent || "Stop / withdraw participation";
  state.withdrawing = true;
  if (button) {
    button.disabled = true;
    button.textContent = "Preparing withdrawal…";
  }
  try {
    if (state.practiceMode || state.testMode) {
      if (button) button.textContent = "Stopping…";
      await api("/api/withdraw", {method: "POST", body: "{}"});
    } else {
      const participantCode = window.prompt(
        "Enter your full participant code to confirm permanent withdrawal:",
      );
      if (participantCode === null) {
        state.withdrawing = false;
        if (button) {
          button.disabled = false;
          button.textContent = idleLabel;
        }
        return;
      }
      const consentVersion = await resolveConsentVersion();
      if (button) button.textContent = "Withdrawing…";
      await api("/api/withdraw", {
        method: "POST",
        body: JSON.stringify({
          participant_code: participantCode.trim(),
          consent_version: consentVersion,
        }),
      });
    }
    sessionStorage.removeItem("human-cbu-session");
    localStorage.removeItem("human-cbu-consent-version");
    if (!state.practiceMode) localStorage.removeItem(participantCodeStorageKey);
    if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
    state.token = null;
    state.consentVersion = null;
    state.participantInformation = null;
    state.imageObjectUrl = null;
    renderWaiting(
      state.practiceMode
        ? "The rehearsal stopped. No answer values were retained."
        : state.testMode
          ? "The participant-interface test stopped. No answer values were retained."
        : "Your withdrawal was recorded. The approved withdrawal policy now applies.",
      {
        title: state.practiceMode
          ? "Rehearsal stopped."
          : state.testMode
            ? "Test stopped."
            : "Withdrawal recorded.",
        allowWithdrawal: false,
      },
    );
  } catch (error) {
    if (error.status === 401) {
      recoverExpiredSession(
        "Your session expired before withdrawal was recorded. Sign in again with your participant code, then choose Stop / withdraw again.",
      );
      return;
    }
    state.withdrawing = false;
    if (button) {
      button.disabled = false;
      button.textContent = idleLabel;
    }
    window.alert(error.message);
  }
}

function pauseStudy() {
  sessionStorage.removeItem("human-cbu-session");
  if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
  state.token = null;
  state.bootstrap = null;
  state.task = null;
  state.answers = {};
  state.startedAt = null;
  state.imagePrimaryLocked = false;
  state.imageRevealed = false;
  state.coverageCaption = null;
  state.imageObjectUrl = null;
  state.previousTask = null;
  state.reviewingPrevious = false;
  state.pairTransition = false;
  renderLogin(
    "Paused safely. Every saved response is unchanged. Continue with your participant code whenever you are ready.",
  );
}

function focusUsefulnessStep() {
  requestAnimationFrame(() => {
    const scrollRegion = app.querySelector(".response-scroll");
    if (scrollRegion) scrollRegion.scrollTop = 0;
    const question = app.querySelector("#coverage-question");
    if (question) {
      question.focus({preventScroll: true});
      question.scrollIntoView({block: "center"});
      return;
    }
    app.querySelector("#save-next")?.focus();
  });
}

function measuredResponseElapsedMs(now = performance.now()) {
  if (state.startedAt === null) return 0;
  const openGuideMs =
    state.guideStartedAt === null ? 0 : Math.max(0, now - state.guideStartedAt);
  return Math.max(
    0,
    Math.round(now - state.startedAt - state.guideElapsedMs - openGuideMs),
  );
}

async function saveAndContinue() {
  if (
    state.task.phase === "image" &&
    !["image_unavailable", "prefer_not_to_answer"].includes(state.answers.image_support) &&
    !state.imagePrimaryLocked
  ) {
    const button = app.querySelector("#save-next");
    if (button) {
      button.disabled = true;
      button.textContent = "Locking…";
    }
    try {
      const result = await api(
        `/api/coverage-caption/${encodeURIComponent(state.task.assignment_id)}`,
      );
      if (typeof result.caption !== "string") throw new Error("Paired caption is unavailable.");
      state.coverageCaption = result.caption;
      state.imagePrimaryLocked = true;
      renderTask(state.task, state.bootstrap?.progress || {});
      focusUsefulnessStep();
    } catch (error) {
      if (button) {
        button.disabled = false;
        button.textContent = "Retry lock & compare caption";
      }
      window.alert(error.message);
    }
    return;
  }
  const button = app.querySelector("#save-next");
  const skipButton = app.querySelector("#skip-image-unseen");
  if (button) {
    button.disabled = true;
    button.textContent = "Saving…";
  }
  if (skipButton) {
    skipButton.disabled = true;
    skipButton.textContent = "Skipping…";
  }
  const payload = {
    assignment_id: state.task.assignment_id,
    answers: state.answers,
    elapsed_ms: measuredResponseElapsedMs(),
  };
  const completedTask = {...state.task};
  try {
    const result = await api("/api/annotation", {method: "POST", body: JSON.stringify(payload)});
    state.previousTask = completedTask;
    state.pairTransition = completedTask.phase === "caption" && result.phase === "image";
    state.answers = {};
    state.imagePrimaryLocked = false;
    state.imageRevealed = false;
    state.coverageCaption = null;
    applyStudyResult(result);
  } catch (error) {
    if (button) {
      button.disabled = false;
      button.textContent = `Retry save`;
    }
    if (skipButton) {
      skipButton.disabled = false;
      skipButton.textContent = "Skip without viewing";
    }
    window.alert(error.message);
  }
}

function renderComplete(progress, workExtension = null) {
  const completed = Number(progress?.completed || progress?.overall_completed || 0);
  const canWithdrawSavedResponses = !state.practiceMode && !state.testMode;
  const canExtend = canWithdrawSavedResponses && Boolean(workExtension?.can_extend);
  const extensionBatch = Number(workExtension?.batch_size || 0);
  const extensionMaximum = Number(workExtension?.max_items || 0);
  app.innerHTML = `<main class="login-layout">
    <section class="login-card">
      <div class="brand"><span class="brand-mark">✓</span><span class="brand-copy"><strong>Visual Claim Study</strong><span>${state.practiceMode ? "Author rehearsal · no responses retained" : state.testMode ? "Participant test · no responses retained" : "Responses saved"}</span></span></div>
      <h1>All assigned claims are complete.</h1>
      <p>${state.practiceMode ? "Rehearsal complete. All rehearsal responses were discarded and cannot enter the human-study export." : state.testMode ? "Interface test complete. All temporary test responses will disappear when the server restarts and cannot enter the human-study export." : `Thank you. Your ${completed.toLocaleString()} responses were saved under a pseudonymous study ID.`}</p>
      ${
        canExtend
          ? `<div class="optional-work-card">
              <strong>Want to help a little more?</strong>
              <p>Your required set is finished. You may stop now, or take ${extensionBatch} more image–claim pairs. Extra work is optional and capped at ${extensionMaximum} pairs total.</p>
              <button class="secondary-button" id="extend-work" type="button">Do ${extensionBatch} more</button>
            </div>`
          : ""
      }
      ${
        canWithdrawSavedResponses
          ? `<details class="withdrawal-details">
              <summary>Permanent withdrawal options</summary>
              <p class="storage-note">Use this only if you want all responses saved under this participant code excluded. Closing this page does not withdraw them.</p>
              <button class="secondary-button withdrawal-card-button" id="withdraw-study" type="button">Withdraw my saved responses…</button>
            </details>`
          : ""
      }
    </section>
  </main>`;
  bindStudyExitControls();
  app.querySelector("#extend-work")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Preparing more…";
    try {
      const result = await api("/api/extend-work", {
        method: "POST",
        body: JSON.stringify({}),
      });
      applyStudyResult(result);
    } catch (error) {
      button.disabled = false;
      button.textContent = `Do ${extensionBatch} more`;
      window.alert(error.message);
    }
  });
}

function renderWaiting(
  message,
  {title = "Paused safely.", allowWithdrawal = true} = {},
) {
  app.innerHTML = `<main class="login-layout">
    <section class="login-card" role="status">
      <div class="brand"><span class="brand-mark">C</span><span class="brand-copy"><strong>Visual Claim Study</strong><span>Progress saved</span></span></div>
      <h1>${escapeHtml(title)}</h1>
      <p>${escapeHtml(message)}</p>
      ${allowWithdrawal ? `<div class="status-actions">${studyExitControls()}</div>` : ""}
    </section>
  </main>`;
  bindStudyExitControls();
}

function renderBootstrapError() {
  app.innerHTML = `<main class="login-layout">
    <section class="login-card" id="bootstrap-error" role="alert" aria-labelledby="bootstrap-error-title" tabindex="-1">
      <div class="brand"><span class="brand-mark">!</span><span class="brand-copy"><strong>Visual Claim Study</strong><span>Session still available</span></span></div>
      <h1 id="bootstrap-error-title">The study could not be loaded.</h1>
      <p>Your saved responses have not been changed. Check your connection and retry, or pause safely and return later.</p>
      <div class="status-actions">
        ${studyExitControls()}
        <button class="primary-button" id="retry-bootstrap" type="button">Retry loading ${icons.arrow}</button>
      </div>
    </section>
  </main>`;
  app.querySelector("#retry-bootstrap")?.addEventListener("click", (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Retrying…";
    bootstrap();
  });
  bindStudyExitControls();
  requestAnimationFrame(() => {
    app.querySelector("#bootstrap-error")?.focus({preventScroll: true});
  });
}

function applyStudyResult(result) {
  state.bootstrap = result;
  state.practiceMode = Boolean(result.practice_mode);
  state.testMode = Boolean(result.test_mode);
  if (result.consent_version) {
    state.consentVersion = result.consent_version;
    localStorage.setItem("human-cbu-consent-version", state.consentVersion);
  }
  if (result.requires_login) {
    renderLogin();
  } else if (result.status === "consent_required" || result.consent_required) {
    state.consentVersion =
      result.required_consent_version || result.participant_information?.approved_version || null;
    state.participantInformation = result.participant_information || null;
    if (state.consentVersion) {
      localStorage.setItem("human-cbu-consent-version", state.consentVersion);
    }
    renderConsent(state.consentVersion, state.participantInformation);
  } else if (result.status === "waiting_for_assets") {
    renderWaiting(
      "The next image is still being verified. Please return later with the same participant code.",
      {title: "Waiting safely."},
    );
  } else if (result.status === "waiting_for_peer_labels") {
    renderWaiting(
      "You have completed every currently available caption task. More work will appear after another annotator joins or advances the shared queue.",
      {title: "Waiting for another annotator."},
    );
  } else if (result.status === "paused") {
    renderWaiting(
      "Collection is temporarily paused. Your completed responses remain saved.",
      {title: "Study temporarily paused."},
    );
  } else if (result.status === "closed") {
    renderWaiting(
      "Collection is closed. Your completed responses remain saved.",
      {title: "Study collection is closed."},
    );
  } else if (result.task) {
    const completed = Number(
      result.progress?.completed || result.progress?.overall_completed || 0,
    );
    if (
      sessionStorage.getItem(tutorialStorageKey) === "complete" ||
      completed > 0
    ) {
      if (completed > 0) {
        sessionStorage.setItem(tutorialStorageKey, "complete");
      }
      renderTask(result.task, result.progress);
    } else {
      renderTutorial(result.task, result.progress, {restart: true});
    }
  } else {
    renderComplete(result.progress, result.work_extension);
  }
}

async function bootstrap() {
  try {
    const result = await api("/api/bootstrap");
    applyStudyResult(result);
  } catch (error) {
    if (error.status === 401) {
      await recoverExpiredBootstrapSession();
      return;
    }
    if (state.token) {
      renderBootstrapError();
      return;
    }
    renderLogin(error.message);
  }
}

bootstrap();
