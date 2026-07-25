/* Standings page: renders the server-embedded results JSON as a table,
   accuracy bars, and per-question difficulty. No fetches, no libraries. */

(function () {
  "use strict";

  var data = JSON.parse(document.getElementById("results-data").textContent);
  var root = document.getElementById("results-root");

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function fmt(value, suffix) {
    return value == null ? "—" : value + (suffix || "");
  }

  // ---- accuracy bars ----

  var bars = el("div");
  data.contestants.forEach(function (c) {
    var row = el("div", "acc-bar" + (c.name === "Humans" ? " humans" : ""));
    row.appendChild(el("span", "lane-name", c.name));
    var bar = el("span", "bar");
    var fill = el("span");
    fill.style.width = (c.accuracy || 0) + "%";
    bar.appendChild(fill);
    row.appendChild(bar);
    row.appendChild(el("span", "lane-score", fmt(c.accuracy, "%")));
    bars.appendChild(row);
  });
  root.appendChild(bars);

  // ---- main table ----

  root.appendChild(el("p", "eyebrow section-label", "The numbers"));
  var table = el("table", "standings-table");
  var thead = el("thead");
  var headRow = el("tr");
  ["Runner", "Answers", "Accuracy", "Confidence", "Median time",
   "Conf. when right", "Conf. when wrong"].forEach(function (h) {
    headRow.appendChild(el("th", null, h));
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  var tbody = el("tbody");
  data.contestants.forEach(function (c) {
    var tr = el("tr");
    tr.appendChild(el("td", null, c.name));
    tr.appendChild(el("td", null, fmt(c.n_answers)));
    tr.appendChild(el("td", null, fmt(c.accuracy, "%")));
    tr.appendChild(el("td", null, fmt(c.mean_confidence)));
    tr.appendChild(el("td", null, fmt(c.median_seconds, "s")));
    tr.appendChild(el("td", null, fmt(c.conf_when_correct)));
    tr.appendChild(el("td", null, fmt(c.conf_when_wrong)));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  root.appendChild(table);

  // ---- per-question difficulty (current set) ----

  if (data.per_question && data.per_question.length) {
    root.appendChild(el("p", "eyebrow section-label",
      "This week's questions, by human accuracy"));
    var qt = el("table", "standings-table per-question");
    var qh = el("tr");
    ["Q", "Headline", "Answers", "Human accuracy"].forEach(function (h) {
      qh.appendChild(el("th", null, h));
    });
    var qthead = el("thead");
    qthead.appendChild(qh);
    qt.appendChild(qthead);
    var qb = el("tbody");
    data.per_question.forEach(function (q) {
      var tr = el("tr");
      tr.appendChild(el("td", null, String(q.position)));
      tr.appendChild(el("td", null, q.headline || ""));
      tr.appendChild(el("td", null, String(q.n_human_answers)));
      tr.appendChild(el("td", null, fmt(q.human_accuracy, "%")));
      qb.appendChild(tr);
    });
    qt.appendChild(qb);
    root.appendChild(qt);
  }
})();
