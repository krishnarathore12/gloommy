import { useState } from "react";
import Navbar from "./Navbar";
import { cn } from "@/lib/utils"; // Utility for merging classes

const HoverNavbar = () => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div 
      className="fixed top-0 left-0 right-0 z-50 flex flex-col items-center"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {/* 1. The Invisible Trigger Area */}
      {/* This creates a 20px zone at the top. Hovering this triggers the menu. */}
      <div className="w-full h-4 bg-transparent absolute top-0 z-50" />

      {/* 2. The Navbar Container */}
      <div 
        className={cn(
          "w-full transition-transform duration-300 ease-in-out transform",
          isVisible ? "translate-y-0" : "-translate-y-full"
        )}
      >
        <div className="bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b shadow-sm">
          <Navbar />
        </div>
      </div>
    </div>
  );
};

export default HoverNavbar;