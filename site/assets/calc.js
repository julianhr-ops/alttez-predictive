(function (root) {
  "use strict";

  var VENUES = {
    neutral: "at a neutral venue",
    a_home: "with team A at home",
    b_home: "with team B at home"
  };

  function triple(grid, n, i, j) {
    var o = (i * n + j) * 3;
    return [grid[o], grid[o + 1], grid[o + 2]];
  }

  function matchup(data, i, j, venue) {
    var n = data.teams.length;
    if (i === j || i < 0 || j < 0 || i >= n || j >= n) {
      return null;
    }
    var t;
    if (venue === "neutral") {
      t = triple(data.neutral, n, i, j);
    } else if (venue === "a_home") {
      t = triple(data.home, n, i, j);
    } else if (venue === "b_home") {
      var flipped = triple(data.home, n, j, i);
      t = [flipped[2], flipped[1], flipped[0]];
    } else {
      throw new Error("unknown venue " + venue);
    }
    return { a: t[0], draw: t[1], b: t[2] };
  }

  function fold(text) {
    return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  }

  function filterTeams(teams, query) {
    var q = fold(query.trim());
    var out = [];
    for (var k = 0; k < teams.length; k++) {
      if (q === "" || fold(teams[k].name).indexOf(q) !== -1) {
        out.push(k);
      }
    }
    return out;
  }

  function percent(units) {
    return (units / 100).toFixed(1) + "%";
  }

  var api = { matchup: matchup, filterTeams: filterTeams, percent: percent, VENUES: VENUES };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.AlttezCalc = api;

  if (typeof document === "undefined") {
    return;
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function fillSelect(select, teams, indices, keep) {
    select.textContent = "";
    if (indices.length === 0) {
      var none = el("option", "", "No team matches");
      none.disabled = true;
      none.selected = true;
      none.value = "";
      select.appendChild(none);
      return;
    }
    var found = false;
    indices.forEach(function (k) {
      var opt = el("option", "", teams[k].rank + ". " + teams[k].name);
      opt.value = String(k);
      if (String(k) === keep) {
        opt.selected = true;
        found = true;
      }
      select.appendChild(opt);
    });
    if (!found) {
      select.selectedIndex = 0;
    }
  }

  function start(data) {
    var teams = data.teams;
    var selA = document.getElementById("calc-a");
    var selB = document.getElementById("calc-b");
    var filterA = document.getElementById("calc-a-filter");
    var filterB = document.getElementById("calc-b-filter");
    var out = document.getElementById("calc-result");
    var summary = document.getElementById("calc-summary");
    var venueInputs = document.querySelectorAll("input[name='venue']");

    function venue() {
      for (var k = 0; k < venueInputs.length; k++) {
        if (venueInputs[k].checked) {
          return venueInputs[k].value;
        }
      }
      return "neutral";
    }

    function render() {
      out.textContent = "";
      var i = selA.value === "" ? -1 : parseInt(selA.value, 10);
      var j = selB.value === "" ? -1 : parseInt(selB.value, 10);
      if (i < 0 || j < 0) {
        summary.textContent = "Choose a team on each side.";
        return;
      }
      if (i === j) {
        summary.textContent = "Choose two different teams.";
        return;
      }
      var v = venue();
      var r = matchup(data, i, j, v);
      var nameA = teams[i].name;
      var nameB = teams[j].name;

      summary.textContent =
        nameA + " win " + percent(r.a) + ", draw " + percent(r.draw) + ", " + nameB + " win " +
        percent(r.b) + ", " + VENUES[v] + ".";

      var bar = el("div", "triple");
      bar.setAttribute("aria-hidden", "true");
      [["seg-a", r.a], ["seg-draw", r.draw], ["seg-b", r.b]].forEach(function (seg) {
        var s = el("span", seg[0], seg[1] >= 1200 ? percent(seg[1]) : "");
        s.style.width = (seg[1] / 100) + "%";
        bar.appendChild(s);
      });
      out.appendChild(bar);

      var legend = el("ul", "legend");
      [
        ["seg-a", nameA + " win", percent(r.a)],
        ["seg-draw", "Draw", percent(r.draw)],
        ["seg-b", nameB + " win", percent(r.b)]
      ].forEach(function (row) {
        var li = el("li");
        li.appendChild(el("span", "swatch " + row[0]));
        li.appendChild(document.createTextNode(row[1] + " "));
        li.appendChild(el("strong", "", row[2]));
        legend.appendChild(li);
      });
      out.appendChild(legend);

      out.appendChild(
        el(
          "p",
          "muted small",
          "Elo ratings as of " + data.as_of + ". " + nameA + " " + Math.round(teams[i].rating) + ", " +
            nameB + " " + Math.round(teams[j].rating) + "."
        )
      );
    }

    function bind(filter, select) {
      filter.addEventListener("input", function () {
        fillSelect(select, teams, filterTeams(teams, filter.value), select.value);
        render();
      });
      select.addEventListener("change", render);
    }

    fillSelect(selA, teams, filterTeams(teams, ""), "0");
    fillSelect(selB, teams, filterTeams(teams, ""), "1");
    bind(filterA, selA);
    bind(filterB, selB);
    for (var k = 0; k < venueInputs.length; k++) {
      venueInputs[k].addEventListener("change", render);
    }
    document.getElementById("calc").hidden = false;
    var fallback = document.getElementById("calc-fallback");
    if (fallback) {
      fallback.hidden = true;
    }
    render();
  }

  function fail(message) {
    var fallback = document.getElementById("calc-fallback");
    if (fallback) {
      fallback.textContent = message;
      fallback.hidden = false;
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (!document.getElementById("calc")) {
      return;
    }
    var embedded = document.getElementById("ratings-data");
    if (embedded) {
      try {
        start(JSON.parse(embedded.textContent));
      } catch (error) {
        fail("The rating data could not be read.");
      }
      return;
    }
    fetch("data/soccer_ratings.json")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(start)
      .catch(function () {
        fail("The rating data could not be loaded. Serve this folder over HTTP, see docs/site.md.");
      });
  });
})(typeof window !== "undefined" ? window : globalThis);
