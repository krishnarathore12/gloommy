import { BrowserRouter as Router, Routes, Route, Outlet } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import GenerationPage from './pages/GenerationPage';
import AssetLibraryPage from './pages/AssetLibraryPage';
import Navbar from './components/Navbar'; 
import HoverNavbar from './components/HoverNavbar';

// Layout for Landing Page (Always visible Navbar)
const LayoutWithNavbar = () => {
  return (
    <>
      <Navbar />
      <main className="min-h-screen bg-background text-foreground">
        <Outlet />
      </main>
    </>
  );
};

// Layout for App Pages (Hidden/Hover Navbar)
const LayoutHiddenNavbar = () => {
  return (
    <div className="relative bg-background text-foreground h-screen overflow-hidden">
      <HoverNavbar /> 
      {/* Pages render here, full screen */}
      <Outlet />
    </div>
  );
};

function App() {
  return (
    <Router>
      <Routes>
        {/* Public / Landing */}
        <Route element={<LayoutWithNavbar />}>
          <Route path="/" element={<LandingPage />} />
        </Route>

        {/* App Workspace (Generation & Assets) - Navbar hides on hover */}
        <Route element={<LayoutHiddenNavbar />}>
           <Route path="/generate" element={<GenerationPage />} />
           <Route path="/assets" element={<AssetLibraryPage />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;