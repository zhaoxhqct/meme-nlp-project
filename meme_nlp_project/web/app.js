const form = document.querySelector("#analyze-form");
const input = document.querySelector("#text-input");
const charCount = document.querySelector("#char-count");
const button = document.querySelector("#analyze-button");

const emptyState = document.querySelector("#empty-state");
const loadingState = document.querySelector("#loading-state");
const errorState = document.querySelector("#error-state");
const errorMessage = document.querySelector("#error-message");
const resultContent = document.querySelector("#result-content");

const memeVerdict = document.querySelector("#meme-verdict");
const decisionReason = document.querySelector("#decision-reason");
const sentimentBadge = document.querySelector("#sentiment-badge");
const highlightedText = document.querySelector("#highlighted-text");
const matchList = document.querySelector("#match-list");
const dictionaryEditor = document.querySelector("#dictionary-editor");
const dictionaryForm = document.querySelector("#dictionary-form");
const entryMeme = document.querySelector("#entry-meme");
const entryCategory = document.querySelector("#entry-category");
const entryMeaning = document.querySelector("#entry-meaning");
const dictionaryButton = document.querySelector("#dictionary-button");
const dictionaryFeedback = document.querySelector("#dictionary-feedback");

let currentResult = null;

const SENTIMENT_CLASS = {
  "正向": "is-positive",
  "负向": "is-negative",
};

function updateCharacterCount() {
  charCount.textContent = `${input.value.length} / 300`;
}

function showOnly(element) {
  [emptyState, loadingState, errorState, resultContent].forEach((item) => {
    item.hidden = item !== element;
  });
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderHighlightedText(text, matches) {
  highlightedText.replaceChildren();
  const terms = [...new Set(matches.map((item) => item.meme))]
    .filter(Boolean)
    .sort((left, right) => right.length - left.length);

  if (!terms.length) {
    highlightedText.textContent = text;
    return;
  }

  const pattern = new RegExp(
    `(${terms.map(escapeRegExp).join("|")})`,
    "gi",
  );
  const lowerTerms = new Set(terms.map((term) => term.toLowerCase()));

  text.split(pattern).forEach((part) => {
    if (lowerTerms.has(part.toLowerCase())) {
      const mark = document.createElement("mark");
      mark.textContent = part;
      highlightedText.append(mark);
      return;
    }
    highlightedText.append(document.createTextNode(part));
  });
}

function renderMatches(matches, hasMeme) {
  matchList.replaceChildren();
  if (!matches.length) {
    const message = document.createElement("p");
    message.className = "no-match";
    message.textContent = hasMeme
      ? "模型判断包含热梗，但词典没有命中具体词条。"
      : "词典中没有命中热梗词。";
    matchList.append(message);
    return;
  }

  matches.forEach((match) => {
    const item = document.createElement("article");
    item.className = "match-item";

    const term = document.createElement("span");
    term.className = "match-term";
    term.textContent = match.meme;

    const category = document.createElement("span");
    category.className = "match-category";
    category.textContent = match.category;

    const meaning = document.createElement("p");
    meaning.className = "match-meaning";
    meaning.textContent = match.meaning;

    item.append(term, category, meaning);
    matchList.append(item);
  });
}

function renderResult(data) {
  const hasMeme = data.has_meme === "yes";
  memeVerdict.textContent = hasMeme ? "检测到热梗" : "未检测到热梗";
  memeVerdict.className = `verdict ${hasMeme ? "is-yes" : "is-no"}`;
  const signals = data.novel_signals || [];
  decisionReason.textContent = signals.length
    ? `判定依据：${data.decision_reason}（${signals.join("、")}）`
    : `判定依据：${data.decision_reason || "模型判断"}`;

  sentimentBadge.textContent = hasMeme
    ? (data.sentiment || "未判断")
    : "未触发情绪判断";
  sentimentBadge.className = "sentiment-badge";
  if (hasMeme && SENTIMENT_CLASS[data.sentiment]) {
    sentimentBadge.classList.add(SENTIMENT_CLASS[data.sentiment]);
  }

  renderHighlightedText(data.text, data.matched_memes || []);
  renderMatches(data.matched_memes || [], hasMeme);
  const canAdd = hasMeme && !(data.matched_memes || []).length;
  dictionaryEditor.hidden = !canAdd;
  dictionaryFeedback.textContent = "";
  if (canAdd) {
    entryMeme.value = data.text;
    entryCategory.value = signals.some((signal) =>
      signal.includes("游戏")
    )
      ? "游戏类"
      : "其他";
    entryMeaning.value = "";
  }
  currentResult = data;
  showOnly(resultContent);
}

async function analyzeText(text) {
  button.disabled = true;
  button.textContent = "分析中";
  showOnly(loadingState);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "请求失败。");
    }
    renderResult(data);
  } catch (error) {
    errorMessage.textContent =
      error instanceof Error ? error.message : "请求失败。";
    showOnly(errorState);
  } finally {
    button.disabled = false;
    button.textContent = "开始分析";
  }
}

function analyze(event) {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) {
    input.focus();
    return;
  }
  analyzeText(text);
}

async function addDictionaryEntry(event) {
  event.preventDefault();
  if (!currentResult) {
    return;
  }

  dictionaryButton.disabled = true;
  dictionaryButton.textContent = "提交中";
  dictionaryFeedback.textContent = "";
  try {
    const response = await fetch("/api/dictionary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: currentResult.text,
        meme: entryMeme.value.trim(),
        category: entryCategory.value,
        meaning: entryMeaning.value.trim(),
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "加入词典失败。");
    }
    await analyzeText(currentResult.text);
  } catch (error) {
    dictionaryFeedback.textContent =
      error instanceof Error ? error.message : "加入词典失败。";
  } finally {
    dictionaryButton.disabled = false;
    dictionaryButton.textContent = "加入词典";
  }
}

form.addEventListener("submit", analyze);
dictionaryForm.addEventListener("submit", addDictionaryEntry);
input.addEventListener("input", updateCharacterCount);
updateCharacterCount();
