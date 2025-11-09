import Header from "./Header.jsx";
import FactsBar from "./FactsBar.jsx";
import MapFrame from "./MapFrame.jsx";

function App() {
  return (
    <>
      <Header/>
      <FactsBar /> 
      <main>
        <MapFrame />
      </main>
    </>
  );
}
export default App;