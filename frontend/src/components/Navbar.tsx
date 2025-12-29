import { Link } from "react-router-dom";
import { Button } from "./ui/button";
import { ModeToggle } from "./mode-toggle"; // Import the toggle

const Navbar = () => {
  return (
    <nav className="border-b py-4 bg-background"> {/* Added bg-background */}
      <div className="container mx-auto flex items-center justify-between">
        <Link to="/" className="font-micro text-2xl text-primary">
          Gloommy
        </Link>

        <div className="flex items-center gap-4">
          <Link to="/generate" className="text-sm font-medium ">
          <Button variant="outline">
             Create
          </Button>
           
          </Link>
          <Link to="/assets" className="text-sm font-medium ">
            <Button variant="outline">
             Assets Library
          </Button>
          </Link>
          
          {/* Add the Dark Mode Toggle here */}
          <ModeToggle />
          
          {/* <Button size="sm">Sign In</Button> */}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;