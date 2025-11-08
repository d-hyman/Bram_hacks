// FactsBar.jsx
import React from "react";

const facts = [
  { id: 1, title: "Annual forest loss", value: "—", note: "Add your source" },
  { id: 2, title: "Tropical primary loss share", value: "—", note: "Add your source" },
  { id: 3, title: "Avg. CO₂ from deforestation", value: "—", note: "Add your source" },
  { id: 4, title: "Countries driving most loss", value: "—", note: "Add your source" },
];

export default function FactsBar() {
  return (
    <section className="container">
      <div className="facts-grid" role="list" aria-label="Deforestation facts">
        {facts.map((f) => (
          <article className="fact-card" role="listitem" key={f.id}>
            <div className="fact-title">{f.title}</div>
            <div className="fact-value">{f.value}</div>
            {f.note && <div className="fact-note">{f.note}</div>}
          </article>
        ))}
      </div>
    </section>
  );
}
