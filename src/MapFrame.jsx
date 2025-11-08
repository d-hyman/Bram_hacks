// MapFrame.jsx
function MapFrame() {
  return (
    <section className="container">
      <div className="map-wrap">
        <iframe
          title="Leaf Watch Map"
          src="src/map.html"
          className="map-frame"
          loading="lazy"
        />
      </div>
    </section>
  );
}
export default MapFrame;
