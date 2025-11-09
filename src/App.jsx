import Header from "./Header.jsx";
import CountryYearQuery from "./CountryYearQuery.jsx";
import MapFrame from "./MapFrame.jsx";

function App() {
  return (
    <>
      <Header/>
      <main>
        <CountryYearQuery />
        <MapFrame />
      </main>
    </>
  );
}
export default App;