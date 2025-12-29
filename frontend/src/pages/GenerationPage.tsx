import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Wand2, Download, Image as ImageIcon, Loader2 } from "lucide-react";

// Types
interface RetrievalResult {
  image_base64: string;
  caption: string;
  matches: string[];
  dense_score: number;
}

interface GenerateResponse {
  generated_image_base64: string;
  retrieved_context: RetrievalResult[];
}

const GenerationPage = () => {
  const [apiKey, setApiKey] = useState("");
  const [prompt, setPrompt] = useState("");
  const [aspectRatio, setAspectRatio] = useState("1:1");
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Store the full response (image + context)
  const [result, setResult] = useState<GenerateResponse | null>(null);

  const handleGenerate = async () => {
    if (!apiKey) {
      alert("Please enter your Google API Key.");
      return;
    }

    setIsGenerating(true);
    setResult(null); // Clear previous result

    try {
      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: prompt,
          api_key: apiKey,
        }),
      });

      if (!response.ok) {
        throw new Error("Generation failed. Check API key or backend logs.");
      }

      const data: GenerateResponse = await response.json();
      setResult(data);

    } catch (error) {
      console.error(error);
      alert("Error generating image. Ensure backend is running.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      
      {/* --- LEFT SIDEBAR: CONTROLS --- */}
      <aside className="w-[400px] border-r p-6 flex flex-col gap-6 overflow-y-auto bg-muted/10">
        
        <div>
          <h2 className="text-xl font-bold tracking-wide mb-1">
            Configuration
          </h2>
          <p className="text-sm text-muted-foreground">
            Set up your generation parameters.
          </p>
        </div>

        <Separator />

        {/* 1. API Key Section */}
        <div className="space-y-3">
          <Label htmlFor="api-key" className="text-sm font-medium">
            Nano Banana API Key
          </Label>
          <div className="relative">
             <Input 
              id="api-key"
              type="password" 
              placeholder="Enter Gemini API Key..." 
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="bg-card"
            />
          </div>
          <p className="text-[10px] text-muted-foreground">
            Required for Gemini 2.5 Flash Image Model.
          </p>
        </div>

        <Separator />

        {/* 2. Aspect Ratio (Visual only for now, as backend is set to 1:1 in prompt mostly) */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Aspect Ratio</Label>
          <Select value={aspectRatio} onValueChange={setAspectRatio}>
            <SelectTrigger className="bg-card">
              <SelectValue placeholder="Select ratio" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1:1">Square (1:1)</SelectItem>
              <SelectItem value="16:9">Landscape (16:9)</SelectItem>
              <SelectItem value="3:2">Photo (3:2)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* 3. Prompt Section */}
        <div className="space-y-3 flex-1 flex flex-col">
          <Label htmlFor="prompt" className="text-sm font-medium">
            Image Prompt
          </Label>
          <Textarea 
            id="prompt"
            placeholder="Describe your image..." 
            className="flex-1 min-h-[150px] resize-none bg-card p-4 leading-relaxed"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        {/* Generate Button */}
        <Button 
          size="lg" 
          className="w-full font-semibold shadow-lg shadow-primary/20"
          onClick={handleGenerate}
          disabled={!prompt || isGenerating}
        >
          {isGenerating ? (
             <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Retrieving & Generating...</>
          ) : (
             <><Wand2 className="mr-2 h-4 w-4" /> Generate Image</>
          )}
        </Button>
      </aside>


      {/* --- RIGHT MAIN AREA: CANVAS --- */}
      <main className="flex-1 flex flex-col h-full bg-muted/30 relative overflow-hidden">
        
        {/* Pattern Background */}
        <div className="absolute inset-0 bg-grid-pattern opacity-50 pointer-events-none" />

        {/* Top: Main Generation Area */}
        <div className="flex-1 flex flex-col items-center justify-center p-10 min-h-0">
          <div className="relative z-10 h-full max-h-[800px] aspect-square bg-card border shadow-sm rounded-xl overflow-hidden flex items-center justify-center">
            
            {isGenerating ? (
               <div className="flex flex-col items-center gap-3 text-muted-foreground animate-pulse">
                  <Loader2 className="w-12 h-12 animate-spin text-primary" />
                  <p>Processing RAG Pipeline...</p>
               </div>
            ) : result?.generated_image_base64 ? (
               // SHOW GENERATED IMAGE
               <img 
                 src={`data:image/png;base64,${result.generated_image_base64}`} 
                 alt="Generated output" 
                 className="w-full h-full object-contain"
               />
            ) : (
               // EMPTY STATE
               <div className="p-8 text-center space-y-4 text-muted-foreground">
                 <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                   <ImageIcon className="w-10 h-10 opacity-50" />
                 </div>
                 <h3 className="text-xl font-medium text-foreground">Ready to Create</h3>
                 <p className="max-w-xs mx-auto text-sm">
                   Enter your prompt to start the Cross-Modal retrieval and generation process.
                 </p>
               </div>
            )}

            {/* Download Button overlay */}
            {result?.generated_image_base64 && (
               <Button 
                 variant="secondary" 
                 size="sm"
                 className="absolute bottom-4 right-4 gap-2 shadow-md"
                 onClick={() => {
                    const link = document.createElement('a');
                    link.href = `data:image/png;base64,${result.generated_image_base64}`;
                    link.download = `generated_${Date.now()}.png`;
                    link.click();
                 }}
               >
                 <Download className="w-4 h-4" /> Download
               </Button>
            )}
          </div>
        </div>

        {/* Bottom: RAG Context Strip */}
        {result?.retrieved_context && result.retrieved_context.length > 0 && (
           <div className="h-[220px] bg-background border-t p-4 z-20 shrink-0 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                 <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                    Retrieved Context (Pareto Frontier)
                 </h4>
                 <span className="text-xs text-muted-foreground">{result.retrieved_context.length} refs used</span>
              </div>
              
              <div className="flex gap-4 overflow-x-auto pb-2 h-full">
                 {result.retrieved_context.map((ctx, idx) => (
                    <div key={idx} className="w-[140px] shrink-0 flex flex-col gap-2 group cursor-pointer">
                       <div className="aspect-square rounded-md overflow-hidden border relative">
                          <img 
                             src={`data:image/png;base64,${ctx.image_base64}`} 
                             className="w-full h-full object-cover group-hover:scale-110 transition-transform" 
                          />
                          <div className="absolute top-1 right-1 bg-black/60 text-white text-[10px] px-1.5 rounded">
                             {(ctx.dense_score * 100).toFixed(0)}%
                          </div>
                       </div>
                       <p className="text-[10px] text-muted-foreground leading-tight line-clamp-2" title={ctx.caption}>
                          {ctx.caption}
                       </p>
                    </div>
                 ))}
              </div>
           </div>
        )}

      </main>
    </div>
  );
};

export default GenerationPage;