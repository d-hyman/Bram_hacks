import React from "react";
import leaf from "./assets/leaf.png";
import "./Header.css";

export default function Header() {
  return (
    <header className="site-header">
      <div className="container nav">
        {/* Logo / brand (left) */}
        <a href="/" className="brand" aria-label="Leaf Watch home">
          <span className="brand-text">Leaf Watch</span>
        </a>

        {/* Actions (right) */}
        <nav className="nav-actions" aria-label="Primary">
          <a className="btn btn-about" href="#about">About</a>
        </nav>
      </div>
    </header>
  );
}