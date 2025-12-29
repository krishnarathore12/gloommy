import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Image as ImageIcon, Wand2 } from "lucide-react";

const LandingPage = () => {
  return (
    <div className="relative min-h-screen overflow-hidden bg-background">
      {/* 1. Background Elements: Dot Grid & Gradient Blob */}
      <div className="absolute inset-0 pointer-events-none bg-grid-pattern [mask-image:linear-gradient(to_bottom,white,transparent)]" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-primary/20 blur-[120px] rounded-full opacity-50" />

      <div className="container relative mx-auto px-4 pt-16 pb-20 flex flex-col items-center text-center">
        

        {/* 3. Hero Typography */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-3xl md:text-5xl font-bold tracking-tight mb-6 max-w-4xl"
        >
          Boot up the past. <br />Generate the {" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-purple-500 to-blue-600">
            future.
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-xl text-muted-foreground mb-10 max-w-2xl"
        >
          Think of it as Pinterest for your AI. Curate a library of inspiration and use your assets to steer the style, tone, and composition of every generation.
        </motion.p>

        {/* 4. Call to Action Buttons (Standard Size) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="flex flex-col sm:flex-row gap-4 mb-20"
        >
          <Link to="/generate">
            {/* Removed size="lg", h-12, px-8, text-lg */}
            <Button variant="outline" className="gap-2">
              <Wand2 className="w-4 h-4" />
              Start Creating
            </Button>
          </Link>
          
          <Link to="/assets">
             {/* Removed size="lg", h-12, px-8, text-lg */}
            <Button variant="outline" className="gap-2">
              Assets Library <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
        </motion.div>

        {/* 5. Visual "Hero" Element (Smaller Version) */}
        {/* Container height reduced from h-[400px] to h-[300px] */}
        <div className="relative w-full max-w-4xl mx-auto h-[300px] perspective-1000">
           
           {/* Card 1: Left Tilted */}
           {/* Card 1: Left Tilted */}
<motion.div 
  initial={{ opacity: 0, rotate: -15, x: -50 }}
  animate={{ opacity: 1, rotate: -6, x: 0 }}
  transition={{ duration: 0.8, delay: 0.4 }}
  /* CHANGED: md:left-24 -> md:left-[20%] */
  className="absolute left-8 md:left-[20%] top-8 w-48 h-64 bg-card border rounded-xl shadow-xl p-3 rotate-[-6deg] z-10 hidden md:block"
>
             <div className="w-full h-32 bg-muted rounded mb-3 flex items-center justify-center overflow-hidden">
                <span className="text-2xl">🎨</span> 
             </div>
             <div className="h-2 w-3/4 bg-muted-foreground/20 rounded mb-2" />
             <div className="h-2 w-1/2 bg-muted-foreground/20 rounded" />
           </motion.div>

           {/* Card 2: Center Main */}
           <motion.div 
             initial={{ opacity: 0, y: 50 }}
             animate={{ opacity: 1, y: 0 }}
             transition={{ duration: 0.8, delay: 0.5 }}
             /* CHANGED: w-80/96 h-96 -> w-64 md:w-72 h-72 */
             className="absolute left-1/2 -translate-x-1/2 top-0 w-64 md:w-72 h-72 bg-card border border-primary/20 rounded-xl shadow-2xl shadow-primary/10 p-4 z-20 flex flex-col"
           >
              <div className="flex items-center gap-1.5 mb-3 border-b pb-2">
                 <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
                 <div className="w-2.5 h-2.5 rounded-full bg-yellow-400" />
                 <div className="w-2.5 h-2.5 rounded-full bg-green-400" />
              </div>
              <div className="flex-1 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded flex items-center justify-center text-white">
                 <ImageIcon className="w-10 h-10 opacity-50" />
              </div>
              <div className="mt-3 p-2 bg-muted/50 rounded text-[10px] text-muted-foreground font-mono truncate">
                 Prompt: A cyberpunk city...
              </div>
           </motion.div>

           {/* Card 3: Right Tilted */}
           {/* Card 3: Right Tilted */}
<motion.div 
  initial={{ opacity: 0, rotate: 15, x: 50 }}
  animate={{ opacity: 1, rotate: 6, x: 0 }}
  transition={{ duration: 0.8, delay: 0.6 }}
  /* CHANGED: md:right-24 -> md:right-[20%] */
  className="absolute right-8 md:right-[20%] top-8 w-48 h-64 bg-card border rounded-xl shadow-xl p-3 rotate-[6deg] z-10 hidden md:block"
>
             <div className="w-full h-32 bg-muted rounded mb-3 flex items-center justify-center">
                <span className="text-2xl">🚀</span>
             </div>
             <div className="h-2 w-3/4 bg-muted-foreground/20 rounded mb-2" />
             <div className="h-2 w-1/2 bg-muted-foreground/20 rounded" />
           </motion.div>
        </div>

      </div>
    </div>
  );
};

export default LandingPage;