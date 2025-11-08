import Header from "./Header.jsx";
import FactsBar from "./FactsBar.jsx";
import MapFrame from "./MapFrame.jsx";

function App() {
  return (
    <>
      <Header/>
      <FactsBar />   {/* ⬅️ new grid of cards above the map */}
      <main>
        <MapFrame />
      </main>
    </>
  );
}
export default App;