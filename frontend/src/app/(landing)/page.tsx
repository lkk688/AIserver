"use client";

import Link from "next/link";
import { useState } from "react";
import { X } from "lucide-react";
import { useRouter } from "next/navigation";
import landingData from "@/data/landing.json";

function LoginModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const router = useRouter();
  
  if (!isOpen) return null;

  const handleLogin = () => {
    router.push("/dashboard");
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md overflow-hidden animate-in fade-in zoom-in duration-200">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Sign In</h2>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
              <X className="w-6 h-6" />
            </button>
          </div>
          
          <div className="text-center mb-8">
            <h3 className="text-lg font-semibold text-gray-800">Welcome to AI-OCR Pro</h3>
            <p className="text-sm text-gray-500">Sign in to access your dashboard</p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
              <input 
                type="email" 
                defaultValue="admin@gmail.com"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <input 
                type="password" 
                defaultValue="*****"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>
            
            <div className="flex justify-between text-sm">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="rounded text-blue-600" />
                <span className="text-gray-600">Remember me</span>
              </label>
              <a href="#" className="text-blue-600 hover:underline">Forgot password?</a>
            </div>

            <button 
              onClick={handleLogin}
              className="w-full bg-blue-600 text-white py-2.5 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Sign In
            </button>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white text-gray-500">Or continue with</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <button className="flex items-center justify-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                <span className="font-medium text-sm">Google</span>
              </button>
              <button className="flex items-center justify-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                <span className="font-medium text-sm">Microsoft</span>
              </button>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 px-6 py-4 text-center text-sm text-gray-500">
          Don't have an account? <a href="#" className="text-blue-600 font-medium hover:underline">Create Account</a>
        </div>
      </div>
    </div>
  );
}

export default function LandingPage() {
  const [isLoginOpen, setIsLoginOpen] = useState(false);

  return (
    <div className="min-h-screen bg-white font-sans text-gray-900">
      {/* Navbar */}
      <nav className="absolute top-0 left-0 w-full z-10 px-8 py-6 flex justify-between items-center text-white">
        <div className="font-bold text-xl">AI-OCR Pro</div>
        <div className="flex items-center gap-8 text-sm font-medium">
          <Link href="#features" className="hover:text-blue-200 transition-colors">Features</Link>
          <Link href="#pricing" className="hover:text-blue-200 transition-colors">Pricing</Link>
          <Link href="#testimonials" className="hover:text-blue-200 transition-colors">Testimonials</Link>
          <button onClick={() => setIsLoginOpen(true)} className="hover:text-blue-200 transition-colors">Sign In</button>
          <button className="bg-blue-600 px-5 py-2 rounded-md hover:bg-blue-700 transition-colors">Get Started</button>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="bg-[#2b1c68] text-white pt-32 pb-24 px-8 overflow-hidden relative">
        {/* Background blobs or gradient could go here */}
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8 animate-in slide-in-from-left duration-700">
            <h1 className="text-5xl font-bold leading-tight">{landingData.hero.title}</h1>
            <p className="text-lg text-blue-100 max-w-lg">{landingData.hero.subtitle}</p>
            <div className="flex gap-4">
              <button className="bg-white text-[#2b1c68] px-8 py-3 rounded-md font-semibold hover:bg-gray-100 transition-colors">
                {landingData.hero.ctaPrimary}
              </button>
              <button className="border border-white/30 text-white px-8 py-3 rounded-md font-semibold hover:bg-white/10 transition-colors">
                {landingData.hero.ctaSecondary}
              </button>
            </div>
          </div>
          <div className="relative animate-in slide-in-from-right duration-700 delay-200">
            {/* Using a div as placeholder for image to avoid Next.js Image config issues for external URL for now */}
            <div className="rounded-xl overflow-hidden shadow-2xl border border-white/10 bg-white/5 backdrop-blur-sm p-2">
               <img 
                 src={landingData.hero.image} 
                 alt="Dashboard Preview" 
                 className="rounded-lg w-full h-auto object-cover"
               />
               <div className="absolute bottom-6 left-6 right-6 bg-white/90 backdrop-blur-md p-4 rounded-lg shadow-lg text-gray-900 flex items-center justify-between">
                 <div>
                   <p className="font-semibold text-sm">AI Processing Complete</p>
                   <p className="text-xs text-gray-500">99.7% accuracy achieved</p>
                 </div>
                 <button className="bg-[#2b1c68] text-white text-xs px-4 py-2 rounded">Download Results</button>
               </div>
            </div>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section id="features" className="py-24 bg-gray-50 px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <h2 className="text-3xl font-bold text-gray-900">{landingData.features.title}</h2>
            <p className="text-gray-600">{landingData.features.subtitle}</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {landingData.features.items.map((feature, idx) => (
              <div key={idx} className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow border border-gray-100">
                <div className={`w-12 h-12 rounded-full ${feature.color} mb-6`}></div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-24 px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 space-y-4">
            <h2 className="text-3xl font-bold text-gray-900">{landingData.pricing.title}</h2>
            <p className="text-gray-600">{landingData.pricing.subtitle}</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {landingData.pricing.plans.map((plan, idx) => (
              <div 
                key={idx} 
                className={`p-8 rounded-2xl border ${plan.highlight ? 'border-blue-600 shadow-xl ring-1 ring-blue-600 relative' : 'border-gray-200 shadow-sm'}`}
              >
                {plan.highlight && (
                  <span className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-blue-600 text-white text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wide">
                    Most Popular
                  </span>
                )}
                <h3 className="text-lg font-bold text-gray-900 mb-2">{plan.name}</h3>
                <div className="flex items-baseline gap-1 mb-4">
                  <span className="text-4xl font-bold text-gray-900">{plan.price}</span>
                  <span className="text-gray-500">{plan.period}</span>
                </div>
                <p className="text-sm text-gray-500 mb-6">{plan.description}</p>
                <button 
                  className={`w-full py-3 rounded-lg font-semibold mb-8 transition-colors ${
                    plan.highlight 
                      ? 'bg-blue-600 text-white hover:bg-blue-700' 
                      : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                  }`}
                >
                  {plan.cta}
                </button>
                <ul className="space-y-3">
                  {plan.features.map((feat, fIdx) => (
                    <li key={fIdx} className="flex items-center gap-3 text-sm text-gray-600">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-600"></div>
                      {feat}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="py-24 bg-gray-50 px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 space-y-4">
            <h2 className="text-3xl font-bold text-gray-900">{landingData.testimonials.title}</h2>
            <p className="text-gray-600">{landingData.testimonials.subtitle}</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {landingData.testimonials.items.map((item, idx) => (
              <div key={idx} className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-center gap-4 mb-6">
                  <img src={item.avatar} alt={item.name} className="w-12 h-12 rounded-full object-cover" />
                  <div>
                    <h4 className="font-bold text-gray-900">{item.name}</h4>
                    <p className="text-xs text-gray-500">{item.role}</p>
                  </div>
                </div>
                <p className="text-gray-600 italic">"{item.quote}"</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA / Footer */}
      <footer className="bg-[#1a103c] text-white pt-20 pb-10 px-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center mb-20 gap-8">
            <div className="max-w-2xl">
              <h2 className="text-3xl font-bold mb-4">{landingData.footer.ctaTitle}</h2>
              <p className="text-blue-200">{landingData.footer.ctaSubtitle}</p>
            </div>
            <div className="flex gap-4">
              <button className="bg-blue-600 px-8 py-3 rounded-md font-semibold hover:bg-blue-700 transition-colors">
                Start Free Trial
              </button>
              <button className="border border-white/30 px-8 py-3 rounded-md font-semibold hover:bg-white/10 transition-colors">
                Contact Sales
              </button>
            </div>
          </div>

          <div className="border-t border-white/10 pt-12 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-8">
            <div className="col-span-2">
              <h3 className="font-bold text-xl mb-4">AI-OCR Pro</h3>
              <p className="text-sm text-gray-400 max-w-xs">
                Advanced document processing powered by artificial intelligence for unparalleled accuracy.
              </p>
            </div>
            {landingData.footer.links.map((section, idx) => (
              <div key={idx}>
                <h4 className="font-bold text-sm uppercase tracking-wider text-gray-400 mb-4">{section.title}</h4>
                <ul className="space-y-2">
                  {section.items.map((link, lIdx) => (
                    <li key={lIdx}>
                      <a href="#" className="text-sm text-gray-300 hover:text-white transition-colors">{link}</a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          
          <div className="border-t border-white/10 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-gray-500">
            <p>© 2025 AI-OCR Pro. All rights reserved.</p>
            <div className="flex gap-6">
              <a href="#" className="hover:text-white">Privacy Policy</a>
              <a href="#" className="hover:text-white">Terms of Service</a>
              <a href="#" className="hover:text-white">Cookie Policy</a>
            </div>
          </div>
        </div>
      </footer>

      {/* Login Modal */}
      <LoginModal isOpen={isLoginOpen} onClose={() => setIsLoginOpen(false)} />
    </div>
  );
}
