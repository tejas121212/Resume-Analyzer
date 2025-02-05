import PyPDF2
import pytesseract
from PIL import Image
import re
import spacy
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class ResumeAnalyzer:
    def __init__(self):
        # Load the English language model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define expected sections and keywords
        self.expected_sections = {
            # 'contact_info': ['email', 'phone', 'address', 'linkedin'],
            'education': ['university', 'college', 'degree', 'gpa', 'graduation'],
            'experience': ['work', 'experience', 'employment', 'job', 'intern'],
            'skills': ['skills', 'technologies', 'programming', 'languages'],
            'projects': ['projects', 'achievements', 'portfolio'],
        }
        
        # Keywords that indicate strong content
        self.action_verbs = [
            'achieved', 'developed', 'implemented', 'created', 'managed',
            'led', 'designed', 'improved', 'increased', 'reduced'
        ]

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting image text: {str(e)}"

    def analyze_content(self, text):
        """Analyze the content of the resume"""
        doc = self.nlp(text.lower())
        
        # Initialize scores dictionary
        scores = {
            'section_scores': {},
            'content_scores': {},
            'overall_score': 0
        }
        
        # Check for presence of each section
        for section, keywords in self.expected_sections.items():
            section_score = 0
            for keyword in keywords:
                if keyword in text.lower():
                    section_score += 1
            scores['section_scores'][section] = min(100, (section_score / len(keywords)) * 100)
        
        # Analyze content quality
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Check for action verbs
        action_verb_count = sum(1 for verb in self.action_verbs if verb in text.lower())
        scores['content_scores']['action_verbs'] = min(100, (action_verb_count / 10) * 100)
        
        # Check for quantifiable achievements
        number_pattern = r'\d+'
        numbers_found = len(re.findall(number_pattern, text))
        scores['content_scores']['quantifiable_achievements'] = min(100, (numbers_found / 5) * 100)
        
        # Calculate overall score
        section_avg = sum(scores['section_scores'].values()) / len(scores['section_scores'])
        content_avg = sum(scores['content_scores'].values()) / len(scores['content_scores'])
        scores['overall_score'] = (section_avg * 0.6) + (content_avg * 0.4)
        
        return scores

    def get_recommendations(self, scores):
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        # Section-based recommendations
        for section, score in scores['section_scores'].items():
            if score < 70:
                recommendations.append(f"Strengthen your {section.replace('_', ' ')} section. "
                                    f"Consider adding more {', '.join(self.expected_sections[section])}.")
        
        # Content-based recommendations
        if scores['content_scores']['action_verbs'] < 70:
            recommendations.append("Use more action verbs to describe your experiences. "
                                "Examples: achieved, developed, implemented, etc.")
        
        if scores['content_scores']['quantifiable_achievements'] < 70:
            recommendations.append("Add more quantifiable achievements. Use numbers and percentages "
                                "to demonstrate your impact.")
        
        return recommendations

    def analyze_resume(self, file_path):
        """Main function to analyze resume"""
        # Determine file type and extract text
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = self.extract_text_from_image(file_path)
        else:
            return "Unsupported file format. Please provide a PDF or image file."
        
        if text.startswith("Error"):
            return text
        
        # Analyze the resume
        scores = self.analyze_content(text)
        recommendations = self.get_recommendations(scores)
        
        return {
            'scores': scores,
            'recommendations': recommendations
        }

def main():
    # Example usage
    analyzer = ResumeAnalyzer()
    
    # Get file path from user
    file_path = input("Please enter the path to your resume (PDF or image): ")
    
    if not os.path.exists(file_path):
        print("File not found. Please check the file path.")
        return
    
    # Analyze resume
    result = analyzer.analyze_resume(file_path)
    
    # Print results
    print("\n=== Resume Analysis Results ===")
    print(f"\nOverall Score: {result['scores']['overall_score']:.2f}/100")
    
    print("\nSection Scores:")
    for section, score in result['scores']['section_scores'].items():
        print(f"{section.replace('_', ' ').title()}: {score:.2f}/100")
    
    print("\nContent Scores:")
    for metric, score in result['scores']['content_scores'].items():
        print(f"{metric.replace('_', ' ').title()}: {score:.2f}/100")
    
    print("\nRecommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()