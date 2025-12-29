import { useState, useEffect, useRef } from "react";
import Masonry from "react-masonry-css";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { UploadCloud, Link as LinkIcon, FileArchive, Plus, Image as ImageIcon, Loader2 } from "lucide-react";

// Types matching Backend Response
interface ImageItem {
  id: string;
  caption: string;
  image_base64: string;
}

const AssetLibraryPage = () => {
  const [dragActive, setDragActive] = useState(false);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [urlInput, setUrlInput] = useState("");
  const [captionInput, setCaptionInput] = useState("");
  const [isUrlUploading, setIsUrlUploading] = useState(false);

  // 1. Fetch Images on Load
  const fetchImages = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/images?page=1&limit=50");
      const data = await res.json();
      setImages(data.images);
    } catch (err) {
      console.error("Failed to fetch images:", err);
    }
  };

  useEffect(() => {
    fetchImages();
  }, []);

  // 2. Handle File Upload (Zip)
  const handleFileUpload = async (file: File) => {
    if (!file) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/ingest", {
        method: "POST",
        body: formData,
      });
      
      if (res.ok) {
        // Refresh the gallery after successful upload
        await fetchImages();
      } else {
        console.error("Upload failed");
      }
    } catch (err) {
      console.error("Error uploading:", err);
    } finally {
      setIsUploading(false);
    }
  };

  // Drag & Drop Handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };
  const handleUrlUpload = async () => {
    if (!urlInput || !captionInput) {
      alert("Please provide both an Image URL and a Caption.");
      return;
    }

    setIsUrlUploading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/ingest-url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: urlInput,
          caption: captionInput,
        }),
      });

      if (res.ok) {
        // Clear inputs and refresh gallery
        setUrlInput("");
        setCaptionInput("");
        await fetchImages();
      } else {
        const err = await res.json();
        alert(`Error: ${err.detail}`);
      }
    } catch (err) {
      console.error("URL Upload Error:", err);
      alert("Failed to connect to backend.");
    } finally {
      setIsUrlUploading(false);
    }
  };

  const breakpointColumnsObj = {
    default: 5,
    1280: 4,
    1024: 3,
    768: 2,
    640: 1,
  };

  return (
    <div className="h-full w-full overflow-y-auto bg-background p-8 pt-16">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-wide text-primary">Asset Library</h1>
            <p className="text-muted-foreground mt-1">Manage your creative assets and reference materials.</p>
          </div>
          <div className="text-sm text-muted-foreground">
            Total Assets: {images.length}
          </div>
        </div>

        {/* Upload Station */}
        <div className="grid md:grid-cols-[1fr_300px] gap-6">
          <Tabs defaultValue="zip" className="w-full">
            <TabsList className="grid w-full max-w-[400px] grid-cols-2 mb-4">
              <TabsTrigger value="zip">Upload ZIP</TabsTrigger>
              <TabsTrigger value="link">Add Link</TabsTrigger>
            </TabsList>
            
            {/* ZIP UPLOAD TAB */}
            <TabsContent value="zip">
              <Card className="border-2 border-dashed border-muted-foreground/25 bg-muted/5 shadow-none">
                <CardContent 
                  className={`flex flex-col items-center justify-center h-[250px] transition-colors ${dragActive ? "bg-primary/5 border-primary" : ""}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  {isUploading ? (
                    <div className="flex flex-col items-center animate-pulse">
                      <Loader2 className="w-10 h-10 text-primary animate-spin mb-4" />
                      <p className="text-sm font-medium">Processing Images & Embeddings...</p>
                      <p className="text-xs text-muted-foreground mt-1">This uses GPU, please wait.</p>
                    </div>
                  ) : (
                    <>
                      <div className="w-16 h-16 bg-background rounded-full flex items-center justify-center mb-4">
                        <UploadCloud className="w-8 h-8 text-primary" />
                      </div>
                      <h3 className="text-lg font-semibold mb-1">Drag & drop your ZIP file</h3>
                      <p className="text-sm text-muted-foreground mb-6">Contains images & metadata.json</p>
                      
                      <Input 
                        type="file" 
                        accept=".zip,.rar,.7z" 
                        className="hidden" 
                        id="file-upload"
                        ref={fileInputRef}
                        onChange={onFileSelect}
                      />
                      <Button variant="secondary" onClick={() => fileInputRef.current?.click()}>
                        <FileArchive className="w-4 h-4 mr-2" /> Select File
                      </Button>
                    </>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* LINK TAB */}
            <TabsContent value="link">
              <Card>
                <CardHeader>
                  <CardTitle>Import from URL</CardTitle>
                  <CardDescription>Add a direct link to an image.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* URL Input */}
                  <div className="space-y-2">
                    <Label htmlFor="url">Image URL</Label>
                    <Input 
                      id="url" 
                      placeholder="https://example.com/image.jpg" 
                      value={urlInput}
                      onChange={(e) => setUrlInput(e.target.value)}
                    />
                  </div>
                  
                  {/* Caption Input */}
                  <div className="space-y-2">
                    <Label htmlFor="caption">Caption</Label>
                    <Input 
                      id="caption" 
                      placeholder="Describe this asset..." 
                      value={captionInput}
                      onChange={(e) => setCaptionInput(e.target.value)}
                    />
                  </div>

                  {/* Import Button */}
                  <Button 
                    className="w-full" 
                    onClick={handleUrlUpload}
                    disabled={isUrlUploading}
                  >
                    {isUrlUploading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Importing...
                      </>
                    ) : (
                      "Import Asset"
                    )}
                  </Button>

                  <div className="pt-2">
                     <div className="p-4 bg-muted/30 rounded-lg text-xs text-muted-foreground flex gap-2">
                        <LinkIcon className="w-4 h-4 shrink-0" />
                        Supported: Direct links to .jpg, .png, .webp
                     </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Gallery Grid */}
        <div className="pt-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Database Gallery</h2>
            <Button variant="ghost" size="sm" className="gap-1 text-muted-foreground hover:text-primary" onClick={fetchImages}>
               Refresh
            </Button>
          </div>
          
          <Masonry
            breakpointCols={breakpointColumnsObj}
            className="flex w-auto -ml-4" 
            columnClassName="pl-4 bg-clip-padding"
          >
            {images.map((item) => (
              <div key={item.id} className="mb-4">
                <div className="group relative w-full bg-muted/20 border rounded-xl overflow-hidden shadow-sm hover:shadow-lg transition-all cursor-pointer">
                  
                  {/* Image Display */}
                  <img 
                    src={`data:image/png;base64,${item.image_base64}`} 
                    alt={item.caption}
                    className="w-full h-auto object-cover"
                    loading="lazy"
                  />
                  
                  {/* Hover Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 p-5 flex flex-col justify-end">
                    <h4 className="text-white font-semibold leading-tight mb-1 drop-shadow-md line-clamp-1">
                      {item.id}
                    </h4>
                    <p className="text-white/90 text-xs leading-relaxed line-clamp-2 mb-2">
                      {item.caption}
                    </p>
                  </div>
                  
                </div>
              </div>
            ))}
          </Masonry>
          
          {images.length === 0 && !isUploading && (
             <div className="text-center py-20 text-muted-foreground">
                <ImageIcon className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <p>No images found. Upload a zip file to populate the database.</p>
             </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default AssetLibraryPage;