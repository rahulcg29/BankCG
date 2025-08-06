import json
import re
import random
from datetime import datetime, timedelta
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional, Union, Any
import ollama
from difflib import get_close_matches
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from io import BytesIO
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from enum import Enum

# Load English language model for NLP
try:
    nlp = spacy.load('en_core_web_sm')
except:
    st.error("Spacy model 'en_core_web_sm' not found. Please install it.")
    st.stop()

# Load the bank data with error handling
try:
    with open('dummydata.json', 'r') as f:
        BANK_DATA = json.load(f)
except FileNotFoundError:
    st.error("Error: dummydata.json file not found!")
    st.stop()
except json.JSONDecodeError:
    st.error("Error: Invalid JSON format in dummydata.json!")
    st.stop()

# Enhanced bank data structure
class BankAccountType(Enum):
    STUDENT = "student_account"
    NRI = "nri_account"
    SENIOR = "senior_account"
    REGULAR = "regular_savings_account"
    CURRENT = "current_account"
    BUSINESS = "business_account"

class LoanType(Enum):
    HOME = "home_loan"
    PERSONAL = "personal_loan"
    CAR = "car_loan"
    EDUCATION = "education_loan"
    BUSINESS = "business_loan"

class GovernmentScheme(Enum):
    PM_KISAN = "pm_kisan_scheme"
    PM_SVANIDHI = "pm_svanidhi_scheme"
    STANDUP_INDIA = "standup_india_scheme"
    MUDRA = "mudra_loan_scheme"

# Ensure all required keys exist in BANK_DATA
REQUIRED_KEYS = ['users', 'bank_info', 'loan_products', 'government_schemes', 
                'transactions_history', 'bills', 'spending_categories', 'bot_responses',
                'account_info', 'account_requests', 'atm_locations', 'branch_locations']

for key in REQUIRED_KEYS:
    if key not in BANK_DATA:
        if key == 'account_requests':
            BANK_DATA['account_requests'] = []
        elif key == 'users':
            BANK_DATA['users'] = {
                "admin": {
                    "name": "Admin User",
                    "email": "admin@cgbank.com",
                    "phone": "9876543210",
                    "address": "CGBank Head Office",
                    "account_type": "Admin Account",
                    "aadhar_number": "999999999999",
                    "pan_number": "AAAAA9999A",
                    "balance": 100000.0,
                    "account_number": "0000000001",
                    "password": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"  # sha256 of 'admin'
                }
            }
        elif key == 'atm_locations':
            BANK_DATA['atm_locations'] = [
                {"location": "Avinashi Road Main Branch", "address": "174/2 Avinashi Road, Coimbatore", "distance": "0.5km"},
                {"location": "RS Puram Branch", "address": "45, DB Road, RS Puram", "distance": "2.1km"},
                {"location": "Gandhipuram ATM", "address": "Near Gandhipuram Bus Stand", "distance": "3.4km"}
            ]
        elif key == 'branch_locations':
            BANK_DATA['branch_locations'] = [
                {"name": "Main Branch", "address": "174/2 Avinashi Road, Coimbatore", "timings": "9:30 AM - 4:30 PM (Mon-Fri), 9:30 AM - 1:30 PM (Sat)"},
                {"name": "RS Puram Branch", "address": "45, DB Road, RS Puram", "timings": "9:30 AM - 4:30 PM (Mon-Fri), 9:30 AM - 1:30 PM (Sat)"},
                {"name": "Peelamedu Branch", "address": "12, Nehru Street, Peelamedu", "timings": "9:30 AM - 4:30 PM (Mon-Fri)"}
            ]
        else:
            st.error(f"Error: Missing required key '{key}' in dummydata.json!")
            st.stop()

# Enhanced account types with more details
if 'bank_accounts' not in BANK_DATA:
    BANK_DATA['bank_accounts'] = {
        'student_account': {
            'name': 'Student Account',
            'min_balance': 0,
            'interest_rate': 2.5,
            'documents': 'Student ID, Address Proof, Aadhar Card',
            'features': 'Zero balance account, Special education loans, Discounts on student services, Free debit card',
            'eligibility': 'Must be a student aged 18-25 with valid student ID',
            'benefits': ['No minimum balance', 'Free SMS alerts', 'Special loan rates']
        },
        'nri_account': {
            'name': 'NRI Account',
            'min_balance': 5000,
            'interest_rate': 3.5,
            'documents': 'Passport, Visa, Address Proof, PAN Card',
            'features': 'Foreign currency support, Global banking, Higher interest rates, Multi-currency debit card',
            'eligibility': 'Must be an Indian citizen residing abroad',
            'benefits': ['Free international transfers', 'Dual currency debit card', 'Tax advisory services']
        },
        'senior_account': {
            'name': 'Senior Citizen Account',
            'min_balance': 1000,
            'interest_rate': 4.0,
            'documents': 'Age Proof, Address Proof, Aadhar Card',
            'features': 'Higher interest rates, Priority services, Special pension benefits, Free health checkups',
            'eligibility': 'Must be 60 years or older',
            'benefits': ['0.5% extra interest', 'Free demand drafts', 'Priority customer support']
        },
        'regular_savings_account': {
            'name': 'Regular Savings Account',
            'min_balance': 1000,
            'interest_rate': 3.0,
            'documents': 'ID Proof, Address Proof, PAN Card',
            'features': 'Free ATM withdrawals, Online banking, Monthly statements, Mobile banking',
            'eligibility': 'Indian resident aged 18+',
            'benefits': ['Free cheque book', 'Net banking', 'Mobile banking app']
        },
        'current_account': {
            'name': 'Current Account',
            'min_balance': 10000,
            'interest_rate': 0.0,
            'documents': 'Business Proof, Address Proof, PAN Card',
            'features': 'Unlimited transactions, Overdraft facility, Business banking services',
            'eligibility': 'Registered business entity',
            'benefits': ['Free cash deposits', 'Overdraft facility', 'Business loan priority']
        }
    }

# Enhanced loan products
if 'loan_products' not in BANK_DATA:
    BANK_DATA['loan_products'] = {
        'home_loan': {
            'name': 'Home Loan',
            'amount': 'Up to ₹5 Crores',
            'interest': '8.4% - 9.2% p.a.',
            'tenure': 'Up to 30 years',
            'documents': ['Identity proof', 'Address proof', 'Income proof', 'Property documents'],
            'eligibility': ['Minimum age 21 years', 'Stable income source', 'Good credit score'],
            'features': 'Low interest rates, Flexible repayment options, Tax benefits',
            'application': 'Apply online or visit any branch with documents'
        },
        'personal_loan': {
            'name': 'Personal Loan',
            'amount': 'Up to ₹20 Lakhs',
            'interest': '10.5% - 15% p.a.',
            'tenure': 'Up to 5 years',
            'documents': ['Identity proof', 'Address proof', 'Income proof', 'Bank statements'],
            'eligibility': ['Minimum age 21 years', 'Minimum salary ₹25,000', 'Good credit score'],
            'features': 'Quick approval, No collateral required, Flexible tenure',
            'application': 'Apply online with minimal documentation'
        },
        'car_loan': {
            'name': 'Car Loan',
            'amount': 'Up to ₹50 Lakhs',
            'interest': '7.9% - 9.5% p.a.',
            'tenure': 'Up to 7 years',
            'documents': ['Identity proof', 'Address proof', 'Income proof', 'Car quotation'],
            'eligibility': ['Minimum age 21 years', 'Stable income source', 'Good credit score'],
            'features': 'Low processing fees, Quick disbursal, Flexible repayment',
            'application': 'Apply online or at dealership'
        },
        'education_loan': {
            'name': 'Education Loan',
            'amount': 'Up to ₹1.5 Crores',
            'interest': '8.5% - 10.5% p.a.',
            'tenure': 'Up to 15 years',
            'documents': ['Admission letter', 'Fee structure', 'Identity proof', 'Co-applicant details'],
            'eligibility': ['Admission to recognized institution', 'Indian nationality', 'Co-applicant required'],
            'features': 'Moratorium period, Low interest rates, Tax benefits',
            'application': 'Apply with admission proof and fee details'
        }
    }

# Enhanced government schemes
if 'government_schemes' not in BANK_DATA:
    BANK_DATA['government_schemes'] = {
        'pm_kisan_scheme': {
            'name': 'PM Kisan Samman Nidhi',
            'benefits': ['₹6,000 per year in 3 installments', 'Direct benefit transfer', 'No middlemen'],
            'eligibility': 'Small and marginal farmers with landholding up to 2 hectares',
            'documents': ['Land records', 'Aadhar card', 'Bank account details'],
            'application': 'Apply at Common Service Centers or online through PM Kisan portal'
        },
        'pm_svanidhi_scheme': {
            'name': 'PM Street Vendor\'s AtmaNirbhar Nidhi',
            'benefits': ['₹10,000 working capital loan', 'Interest subsidy', 'Digital transactions incentive'],
            'eligibility': 'Street vendors operating before March 24, 2020',
            'documents': ['Vendor certificate', 'Aadhar card', 'Bank account details'],
            'application': 'Apply through Urban Local Bodies or online portal'
        },
        'standup_india_scheme': {
            'name': 'StandUp India Scheme',
            'benefits': ['Loan from ₹10 lakh to ₹1 crore', 'Composite loan for setup', 'Greenfield enterprise support'],
            'eligibility': 'SC/ST and women entrepreneurs for greenfield projects',
            'documents': ['Business plan', 'Identity proof', 'Caste certificate (if applicable)'],
            'application': 'Apply through designated branches with business proposal'
        },
        'mudra_loan_scheme': {
            'name': 'MUDRA Loan Scheme',
            'benefits': ['Loans up to ₹10 lakh', 'No collateral required', 'Support for non-farm income activities'],
            'eligibility': 'Small business units and micro enterprises',
            'documents': ['Business proof', 'Identity proof', 'Bank statements'],
            'application': 'Apply through any bank branch or online portal'
        }
    }

class PDFGenerator:
    """Enhanced PDF report generator with better formatting and security features"""
    
    @staticmethod
    def generate_pdf_report(user_data: Dict[str, Any], report_data: Dict[str, Any]) -> BytesIO:
        """Generate a professional PDF banking report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                              title=f"CGBank Statement - {user_data['name']}",
                              author="CGBank Digital Services")
        
        styles = getSampleStyleSheet()
        elements = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=1,
            spaceAfter=12,
            textColor=colors.HexColor("#1e3c72")
        )
        
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=6,
            textColor=colors.HexColor("#2a5298")
        )
        
        # Add bank logo and header
        elements.append(Paragraph('<img src="https://via.placeholder.com/150x50?text=CGBank" width="150" height="50"/>', 
                                styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph('CGBank - Monthly Statement Report', title_style))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d-%b-%Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 0.5 * inch))
        
        # Confidential notice
        elements.append(Paragraph('<font color="red"><b>CONFIDENTIAL - FOR CUSTOMER USE ONLY</b></font>', 
                                styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Account Holder Information
        elements.append(Paragraph('Account Holder Information', header_style))
        
        import hashlib

        user_info = [
            ["Account Holder", user_data['name']],
            ["Account Number", f"XXXXXX{user_data['account_number'][-4:]}"],
            ["Account Type", user_data['account_type']],
            ["Report Period", f"{report_data['start_date']} to {report_data['end_date']}"],
            ["Current Balance", f"₹{user_data['balance']:,.2f}"],
            ["Customer ID", f"CID-{hashlib.sha256(user_data['name'].encode()).hexdigest()[:8]}"]
        ]

        
        user_table = Table(user_info, colWidths=[2*inch, 4*inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8f9fa")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
        ]))
        
        elements.append(user_table)
        elements.append(Spacer(1, 0.5 * inch))
        
        # Transaction Summary
        elements.append(Paragraph('Transaction Summary', header_style))
        
        summary_data = [
            ["Total Transactions", str(report_data['total_transactions'])],
            ["Total Credit", f"₹{report_data['total_credit']:,.2f}"],
            ["Total Debit", f"₹{report_data['total_debit']:,.2f}"],
            ["Net Change", f"₹{report_data['net_change']:,.2f}"],
            ["Average Daily Balance", f"₹{report_data.get('avg_balance', user_data['balance']):,.2f}"],
            ["Interest Earned", f"₹{report_data.get('interest_earned', 0):,.2f}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8f9fa")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.5 * inch))
        
        # Spending Analysis
        if 'spending_analysis' in report_data:
            elements.append(Paragraph('Spending Analysis', header_style))
            
            spending_data = [["Category", "Amount", "Percentage"]]
            for category, amount in report_data['spending_analysis'].items():
                percentage = (amount / report_data['total_debit']) * 100 if report_data['total_debit'] > 0 else 0
                spending_data.append([
                    category,
                    f"₹{abs(amount):,.2f}",
                    f"{percentage:.1f}%"
                ])
            
            spending_table = Table(spending_data, colWidths=[2*inch, 2*inch, 2*inch])
            spending_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8f9fa")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
            ]))
            
            elements.append(spending_table)
            elements.append(Spacer(1, 0.5 * inch))
        
        # Transaction Details
        elements.append(Paragraph('Transaction Details', header_style))
        
        transaction_data = [["Date", "Description", "Amount", "Balance", "Category"]]
        for txn in report_data['transactions']:
            date_str = txn['date'].strftime('%d-%b-%Y') if isinstance(txn['date'], datetime) else txn['date']
            amount = txn['amount']
            amount_str = f"+₹{amount:,.2f}" if amount > 0 else f"-₹{abs(amount):,.2f}"
            category = txn.get('category', 'Other')
            transaction_data.append([
                date_str,
                txn['description'],
                amount_str,
                f"₹{txn['balance']:,.2f}",
                category
            ])
        
        transaction_table = Table(transaction_data, 
                                colWidths=[1*inch, 2.5*inch, 1.2*inch, 1.2*inch, 1*inch])
        transaction_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8f9fa")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        
        elements.append(transaction_table)
        elements.append(Spacer(1, 0.5 * inch))
        
        # Financial Insights
        if len(report_data['transactions']) > 10:
            elements.append(Paragraph('Financial Insights', header_style))
            
            insights = [
                f"• Your largest credit was ₹{max(t['amount'] for t in report_data['transactions'] if t['amount'] > 0):,.2f}",
                f"• Your largest debit was ₹{abs(min(t['amount'] for t in report_data['transactions'] if t['amount'] < 0)):,.2f}",
                f"• You made {sum(1 for t in report_data['transactions'] if t['amount'] < 0)} debit transactions",
                f"• You received {sum(1 for t in report_data['transactions'] if t['amount'] > 0)} credit transactions"
            ]
            
            for insight in insights:
                elements.append(Paragraph(insight, styles['Normal']))
                elements.append(Spacer(1, 0.2 * inch))
        
        # Footer section
        footer = """
        <para>
        <font size=9>
        <b>Notes:</b><br/>
        1. This is an automatically generated statement. For any discrepancies, please contact CGBank customer support within 7 days.<br/>
        2. Please keep this statement confidential. Destroy it securely when no longer needed.<br/>
        3. Visit www.cgbank.com for digital banking services or download our mobile app.<br/>
        </font>
        </para>
        """
        elements.append(Paragraph(footer, styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Bank contact information
        contact = """
        <para>
        <font size=9>
        <b>CGBank - Coimbatore Trusted Banking Partner</b><br/>
        174/2 Avinashi Road, Coimbatore - 641029<br/>
        Helpline: 1800-123-4506 | Email: support@cgbank.com<br/>
        Working Hours: 9:30 AM - 4:30 PM (Mon-Fri), 9:30 AM - 1:30 PM (Sat)
        </font>
        </para>
        """
        elements.append(Paragraph(contact, ParagraphStyle('Footer', parent=styles['Normal'], alignment=1)))
        
        # Add page number
        def add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.drawString(inch, 0.75 * inch, f"Page {doc.page} | {datetime.now().strftime('%d-%b-%Y %H:%M')}")
            canvas.restoreState()
        
        # Build the PDF with page numbers
        doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
        
        buffer.seek(0)
        return buffer

class FeedbackSystem:
    """Enhanced feedback system with sentiment analysis"""
    
    @staticmethod
    def send_feedback_email(name: str, email: str, rating: int, feedback: str):
        """Send feedback email with sentiment analysis"""
        try:
            # Analyze feedback sentiment
            sentiment = "Neutral"
            doc = nlp(feedback)
            sentiment_score = doc._.polarity if hasattr(doc._, 'polarity') else 0
            
            if sentiment_score > 0.3:
                sentiment = "Positive"
            elif sentiment_score < -0.3:
                sentiment = "Negative"
            
            # Email configuration
            sender_email = "feedback@cgbank.com"
            sender_password = "bankfeedback123"
            receiver_email = "customer.experience@cgbank.com"
            
            # Create message
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = f"New Feedback ({sentiment}) - Rating: {rating}/5"
            
            # Email body with styling
            body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <h2 style="color: #2a5298;">New Customer Feedback Received</h2>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                        <p><strong style="color: #1e3c72;">Name:</strong> {name}</p>
                        <p><strong style="color: #1e3c72;">Email:</strong> {email if email else 'Not provided'}</p>
                        <p><strong style="color: #1e3c72;">Rating:</strong> {"⭐" * rating + "☆" * (5 - rating)}</p>
                        <p><strong style="color: #1e3c72;">Sentiment:</strong> {sentiment} (Score: {sentiment_score:.2f})</p>
                        <p><strong style="color: #1e3c72;">Feedback:</strong></p>
                        <div style="background-color: white; padding: 10px; border-radius: 3px; border-left: 4px solid #2a5298;">
                            {feedback}
                        </div>
                    </div>
                    <p style="margin-top: 20px; font-size: 0.9em; color: #666;">
                        This feedback was submitted through CGBank digital platform.
                    </p>
                </body>
            </html>
            """
            
            message.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, message.as_string())
                
            return True
        except Exception as e:
            st.error(f"Error sending feedback email: {str(e)}")
            return False

class CGBankDatabase:
    """Enhanced database class with transaction categorization and analytics"""
    
    @staticmethod
    def _save_data():
        """Save the current BANK_DATA to the JSON file with backup"""
        try:
            # Create backup
            backup_path = Path('dummydata_backup.json')
            if backup_path.exists():
                backup_path.unlink()
            Path('dummydata.json').rename(backup_path)
            
            # Save new data
            with open('dummydata.json', 'w') as f:
                json.dump(BANK_DATA, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
            try:
                # Restore backup if save failed
                if backup_path.exists():
                    backup_path.rename('dummydata.json')
            except:
                pass
            return False
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using PBKDF2 with HMAC-SHA256"""
        salt = "CGBank_Secure_Salt_Value_2023!"
        iterations = 100000
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations
        ).hex()
    
    @staticmethod
    def get_user(username: str) -> Optional[Dict[str, Any]]:
        """Get user data by username with case-insensitive search and fuzzy matching"""
        username_lower = username.lower()
        
        # First try exact match
        for uname, user_data in BANK_DATA['users'].items():
            if uname.lower() == username_lower:
                return user_data
        
        # Then try fuzzy matching for typos
        matches = get_close_matches(username_lower, [u.lower() for u in BANK_DATA['users'].keys()], n=1, cutoff=0.7)
        if matches:
            matched_username = matches[0]
            for uname, user_data in BANK_DATA['users'].items():
                if uname.lower() == matched_username:
                    return user_data
        
        return None
    
    @staticmethod
    def verify_user(username: str, password: str) -> bool:
        """Verify user credentials with brute force protection"""
        # Simple rate limiting (in a real app, use proper rate limiting)
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
            st.session_state.last_attempt = datetime.now()
        
        if (datetime.now() - st.session_state.last_attempt).seconds < 5:
            st.session_state.login_attempts += 1
        else:
            st.session_state.login_attempts = 1
            st.session_state.last_attempt = datetime.now()
        
        if st.session_state.login_attempts > 3:
            st.error("Too many login attempts. Please try again later.")
            return False
        
        user = CGBankDatabase.get_user(username)
        if not user:
            return False
        
        # Get the hashed password from user data
        stored_password = user.get('password')
        if not stored_password:
            return False
            
        # Hash the provided password and compare
        hashed_password = CGBankDatabase.hash_password(password)
        return stored_password == hashed_password
    
    @staticmethod
    def create_user(username: str, password: str, user_data: Dict[str, Any]) -> bool:
        """Create a new user account with validation"""
        if CGBankDatabase.get_user(username):
            return False  # User already exists
            
        # Validate username
        if not re.match(r'^[a-zA-Z0-9_]{4,20}$', username):
            st.error("Username must be 4-20 characters (letters, numbers, underscores)")
            return False
        
        # Validate password strength
        if len(password) < 8:
            st.error("Password must be at least 8 characters")
            return False
        if not re.search(r'[A-Z]', password):
            st.error("Password must contain at least one uppercase letter")
            return False
        if not re.search(r'[a-z]', password):
            st.error("Password must contain at least one lowercase letter")
            return False
        if not re.search(r'[0-9]', password):
            st.error("Password must contain at least one number")
            return False
        
        # Hash the password before storing
        user_data['password'] = CGBankDatabase.hash_password(password)
        
        # Generate account number
        user_data['account_number'] = str(random.randint(1000000000, 9999999999))
        
        # Set default balance based on account type
        account_type = user_data.get('account_type', 'Regular Savings Account')
        if account_type == 'Student Account':
            user_data['balance'] = 0.0
        elif account_type == 'NRI Account':
            user_data['balance'] = 5000.0
        else:
            user_data['balance'] = 1000.0
        
        # Add security questions
        user_data['security_questions'] = {
            'question1': 'What is your mother\'s maiden name?',
            'question2': 'What was the name of your first pet?',
            'question3': 'In what city were you born?'
        }
        
        # Store the user data
        BANK_DATA['users'][username.lower()] = user_data
        
        # Save to JSON file
        return CGBankDatabase._save_data()
    
    @staticmethod
    def get_bank_info() -> Dict[str, Any]:
        """Get enhanced bank information"""
        info = BANK_DATA['bank_info']
        info['services'] = [
            "Personal Banking",
            "Business Banking",
            "Loans",
            "Investments",
            "Insurance",
            "Digital Banking",
            "NRI Services",
            "Wealth Management"
        ]
        info['atm_locations'] = BANK_DATA.get('atm_locations', [])
        info['branch_locations'] = BANK_DATA.get('branch_locations', [])
        return info
    
    @staticmethod
    def get_loan_products() -> Dict[str, Any]:
        """Get loan products with enhanced details"""
        return BANK_DATA['loan_products']
    
    @staticmethod
    def get_government_schemes() -> Dict[str, Any]:
        """Get government schemes with enhanced details"""
        return BANK_DATA['government_schemes']
    
    @staticmethod
    def get_account_info() -> Dict[str, Any]:
        """Get account types information with proper formatting"""
        return BANK_DATA.get('bank_accounts', {})
    
    @staticmethod
    def get_user_transactions(username: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction history for a user with categorization"""
        user = CGBankDatabase.get_user(username)
        if not user:
            return []
        
        # Get transactions from session state if available
        if 'transactions' in st.session_state and st.session_state.transactions:
            return st.session_state.transactions[:limit]
        
        # Otherwise generate new transactions with categories
        transactions = []
        categories = [
            'Shopping', 'Food & Dining', 'Utilities', 
            'Transfer', 'Salary', 'Entertainment',
            'Healthcare', 'Education', 'Travel'
        ]
        
        for txn in BANK_DATA['transactions_history']:
            # Determine category based on description
            desc = txn['name'].lower()
            if 'salary' in desc:
                category = 'Salary'
            elif any(word in desc for word in ['transfer', 'send', 'received']):
                category = 'Transfer'
            elif any(word in desc for word in ['electric', 'water', 'gas', 'bill']):
                category = 'Utilities'
            elif any(word in desc for word in ['food', 'restaurant', 'coffee', 'dining']):
                category = 'Food & Dining'
            elif any(word in desc for word in ['movie', 'concert', 'game', 'entertain']):
                category = 'Entertainment'
            elif any(word in desc for word in ['medical', 'hospital', 'pharmacy']):
                category = 'Healthcare'
            elif any(word in desc for word in ['school', 'college', 'tuition', 'education']):
                category = 'Education'
            elif any(word in desc for word in ['flight', 'hotel', 'travel', 'vacation']):
                category = 'Travel'
            elif any(word in desc for word in ['amazon', 'flipkart', 'shopping', 'store']):
                category = 'Shopping'
            else:
                category = 'Other'
            
            transaction = {
                'date': datetime.now() - timedelta(days=random.randint(1, 90)),
                'description': txn['name'],
                'amount': txn['amt'],
                'balance': user['balance'] - random.uniform(0, 1000),
                'category': category
            }
            transactions.append(transaction)
        
        # Sort by date and store in session state
        transactions = sorted(transactions, key=lambda x: x['date'], reverse=True)
        st.session_state.transactions = transactions
        return transactions[:limit]
    
    @staticmethod
    def get_user_bills(username: str) -> List[Dict[str, Any]]:
        """Get bills for a user with enhanced data"""
        return BANK_DATA['bills']
    
    @staticmethod
    def get_spending_categories(username: str) -> List[Dict[str, Any]]:
        """Get spending categories with analytics"""
        transactions = CGBankDatabase.get_user_transactions(username)
        if not transactions:
            return []
        
        # Categorize spending
        categories = {}
        for txn in transactions:
            if txn['amount'] < 0:  # Only consider debits
                category = txn.get('category', 'Other')
                categories[category] = categories.get(category, 0) + abs(txn['amount'])
        
        # Convert to list of dicts
        return [{'name': k, 'amount': v} for k, v in categories.items()]
    
    @staticmethod
    def add_transaction(username: str, description: str, amount: float) -> bool:
        """Add a new transaction with validation and categorization"""
        user = CGBankDatabase.get_user(username)
        if not user:
            return False
        
        # Validate amount
        if amount == 0:
            return False
        
        # Get current transactions
        transactions = CGBankDatabase.get_user_transactions(username)
        
        # Determine category
        desc = description.lower()
        if 'salary' in desc:
            category = 'Salary'
        elif any(word in desc for word in ['transfer', 'send', 'received']):
            category = 'Transfer'
        elif any(word in desc for word in ['electric', 'water', 'gas', 'bill']):
            category = 'Utilities'
        elif any(word in desc for word in ['food', 'restaurant', 'coffee', 'dining']):
            category = 'Food & Dining'
        elif any(word in desc for word in ['movie', 'concert', 'game', 'entertain']):
            category = 'Entertainment'
        elif any(word in desc for word in ['medical', 'hospital', 'pharmacy']):
            category = 'Healthcare'
        elif any(word in desc for word in ['school', 'college', 'tuition', 'education']):
            category = 'Education'
        elif any(word in desc for word in ['flight', 'hotel', 'travel', 'vacation']):
            category = 'Travel'
        elif any(word in desc for word in ['amazon', 'flipkart', 'shopping', 'store']):
            category = 'Shopping'
        else:
            category = 'Other'
        
        # Create new transaction
        new_transaction = {
            'date': datetime.now(),
            'description': description,
            'amount': amount,
            'balance': user['balance'] + amount,
            'category': category
        }
        
        # Insert at beginning of list (most recent first)
        transactions.insert(0, new_transaction)
        
        # Update user balance
        user['balance'] += amount
        
        # Add to transactions_history in BANK_DATA
        BANK_DATA['transactions_history'].append({
            'name': description,
            'amt': amount
        })
        
        # Update session state
        st.session_state.transactions = transactions
        
        # Save to JSON file
        return CGBankDatabase._save_data()
    
    @staticmethod
    def add_bill_payment(username: str, bill_name: str, amount: float) -> bool:
        """Add a bill payment transaction with validation"""
        user = CGBankDatabase.get_user(username)
        if not user:
            return False
        
        # Add to transactions
        success = CGBankDatabase.add_transaction(username, f"Bill Payment: {bill_name}", -amount)
        if not success:
            return False
        
        # Remove paid bill from bills list
        BANK_DATA['bills'] = [bill for bill in BANK_DATA['bills'] if bill['name'] != bill_name]
        
        # Save to JSON file
        return CGBankDatabase._save_data()
    
    @staticmethod
    def add_new_bill(username: str, bill_data: Dict[str, Any]) -> bool:
        """Add a new bill with validation"""
        # Validate bill data
        if not all(key in bill_data for key in ['name', 'amount', 'due']):
            return False
        
        # Add to bills list
        BANK_DATA['bills'].append(bill_data)
        
        # Save to JSON file
        return CGBankDatabase._save_data()
    
    @staticmethod
    def update_user_balance(username: str, amount: float) -> bool:
        """Update user balance with validation"""
        user = CGBankDatabase.get_user(username)
        if not user:
            return False
        
        # Validate balance won't go negative (unless it's a credit account)
        if user['balance'] + amount < 0 and user.get('account_type') != 'Current Account':
            return False
        
        user['balance'] += amount
        return CGBankDatabase._save_data()
    
    @staticmethod
    def request_new_account(account_data: Dict[str, Any]) -> bool:
        """Add a new account request with validation"""
        try:
            # Validate required fields
            required_fields = ['name', 'email', 'phone', 'address', 'account_type']
            if not all(field in account_data for field in required_fields):
                return False
            
            # Generate a unique request ID
            request_id = str(uuid.uuid4())
            account_data['request_id'] = request_id
            account_data['status'] = 'Pending'
            account_data['request_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add to account requests
            BANK_DATA['account_requests'].append(account_data)
            
            # Save to JSON file
            return CGBankDatabase._save_data()
        except Exception as e:
            print(f"Error saving account request: {e}")
            return False
    
    @staticmethod
    def get_atm_locations(zipcode: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ATM locations with optional zipcode filtering"""
        atms = BANK_DATA.get('atm_locations', [])
        if zipcode:
            return [atm for atm in atms if atm.get('zipcode') == zipcode]
        return atms
    
    @staticmethod
    def get_branch_locations(zipcode: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get branch locations with optional zipcode filtering"""
        branches = BANK_DATA.get('branch_locations', [])
        if zipcode:
            return [branch for branch in branches if branch.get('zipcode') == zipcode]
        return branches

class RexaBot:
    """Enhanced CGBank intelligent banking assistant with advanced NLP capabilities"""
    
    def __init__(self):
        self.name = "Rexa"
        self.version = "2.1"
        self.persona = {
            "name": "Rexa",
            "role": "Senior Banking Assistant",
            "bank": "CGBank",
            "tone": "professional yet friendly",
            "communication_style": "clear, concise, and helpful",
            "expertise": [
                "account management",
                "fund transfers",
                "loan products",
                "government schemes",
                "financial planning",
                "digital banking"
            ]
        }
        
        # Enhanced service keywords with more natural language variations
        self.service_keywords = {
            'balance_inquiry': ['balance', 'account balance', 'how much money', 'check balance', 
                              'current balance', 'what do i have', 'funds available', 'available balance',
                              'remaining balance', 'account summary', 'show me my money', 'what\'s my balance',
                              'how much is in my account', 'account funds', 'current funds'],
            
            'transaction_history': ['transaction', 'history', 'statement', 'recent transactions', 
                                  'last transactions', 'past payments', 'my spending', 'past expenses',
                                  'payment history', 'transaction details', 'transaction summary',
                                  'show my transactions', 'where has my money gone', 'past purchases',
                                  'account activity', 'account statement', 'bank statement'],
            
            'fund_transfer': ['transfer', 'send money', 'transfer money', 'transfer funds', 
                            'move money', 'pay someone', 'send to friend', 'wire transfer',
                            'remit', 'send cash', 'make a payment', 'pay a contact',
                            'send funds', 'money transfer', 'online payment'],
            
            'bill_payment': ['pay bill', 'bill payment', 'utility bill', 'electricity bill', 
                           'water bill', 'gas bill', 'phone bill', 'internet bill',
                           'credit card bill', 'mobile recharge', 'pay utilities',
                           'clear my bills', 'pay my dues', 'settle bills'],
            
            'bank_info': ['about cgbank', 'bank information', 'bank details', 'what is cgbank', 
                         'bank services', 'products offered', 'bank features', 'branch locations',
                         'contact bank', 'bank timings', 'working hours', 'bank address',
                         'about the bank', 'bank overview', 'bank description'],
            
            'loan_info': ['loan', 'borrow', 'credit', 'home loan', 'personal loan', 'car loan',
                         'education loan', 'interest rates', 'eligibility', 'how to apply',
                         'mortgage', 'vehicle loan', 'student loan', 'business loan',
                         'get a loan', 'loan options', 'loan products', 'loan details'],
            
            'scheme_info': ['government scheme', 'scheme details', 'scheme information', 'subsidy',
                          'farmer benefits', 'women benefits', 'agricultural scheme', 'pm kisan',
                          'mudra loan', 'standup india', 'pm svanidhi', 'government benefit',
                          'subsidy scheme', 'welfare scheme', 'financial aid'],
            
            'account_info': ['account', 'account type', 'student account', 'nri account', 'senior account', 
                            'how to open new acc', 'account types', 'new account', 'create account', 
                            'open account', 'account opening', 'how to create account', 'bank account',
                            'savings account', 'current account', 'account features', 'account benefits'],
            
            'monthly_report': ['monthly report', 'monthly statement', 'monthly summary', 
                             'monthly transactions', 'monthly spending', 'monthly analysis',
                             'report for month', 'statement for month', 'account summary',
                             'financial summary', 'monthly overview', 'spending report'],
            
            'filter_transactions': ['transactions above', 'transactions below', 'transactions between',
                                  'transactions greater than', 'transactions less than', 
                                  'transactions from', 'transactions to', 'transactions in',
                                  'show transactions', 'find transactions', 'search transactions',
                                  'filter my transactions', 'specific transactions', 'transactions by amount',
                                  'transactions by date', 'transactions in range'],
            
            'atm_info': ['atm', 'atm location', 'nearest atm', 'atm near me', 'atm card', 
                        'atm withdrawal limit', 'atm charges', 'atm limit', 'atm pin',
                        'cash machine', 'where can i withdraw', 'find atm', 'atm locator',
                        'cash withdrawal', 'atm services'],
            
            'card_info': ['debit card', 'credit card', 'card details', 'card limit', 
                         'card activation', 'card lost', 'card stolen', 'card block',
                         'card replacement', 'card upgrade', 'new card', 'card services',
                         'card benefits', 'card features', 'card management'],
            
            'customer_support': ['contact support', 'customer service', 'help desk', 'support number',
                               'complaint', 'grievance', 'issue', 'problem', 'help', 'assistance',
                               'talk to someone', 'get help', 'customer care', 'support team',
                               'report problem', 'file complaint'],
            
            'interest_rates': ['interest rate', 'savings rate', 'fd rate', 'fixed deposit rate',
                             'loan rate', 'deposit rate', 'current account rate', 'rd rate',
                             'account interest', 'earning rate', 'yield', 'return on deposit',
                             'bank rates', 'financial rates'],
            
            'security_info': ['security', 'fraud', 'scam', 'phishing', 'safe banking', 
                            'account security', 'secure login', 'two factor authentication',
                            '2fa', 'security tips', 'protect account', 'prevent fraud',
                            'banking safety', 'secure transactions'],
            
            'investment_info': ['investment', 'fd', 'fixed deposit', 'rd', 'recurring deposit',
                              'mutual fund', 'insurance', 'wealth management', 'financial planning',
                              'investment options', 'grow money', 'savings plan', 'retirement plan',
                              'financial future', 'investment advice'],
            
            'financial_advice': ['financial advice', 'money tips', 'saving tips', 'budgeting help',
                               'financial planning', 'money management', 'wealth building',
                               'how to save', 'financial guidance', 'money advice',
                               'financial wellness', 'smart banking']
        }
        
        self.knowledge_base = self._create_knowledge_base()
        self.vectorizer = TfidfVectorizer()
        self._train_similarity_model()
        self._setup_nlp_pipeline()
    
    def _setup_nlp_pipeline(self):
        """Set up custom NLP pipeline components"""
        # Add sentiment analysis if not already present
        if not nlp.has_pipe('sentiment'):
            try:
                from textblob import TextBlob
                def sentiment_component(doc):
                    blob = TextBlob(doc.text)
                    doc._.polarity = blob.sentiment.polarity
                    doc._.subjectivity = blob.sentiment.subjectivity
                    return doc
                
                nlp.add_pipe(sentiment_component, name='sentiment', last=True)
            except:
                pass
    
    def _train_similarity_model(self):
        """Train a TF-IDF model for similarity matching with enhanced corpus"""
        # Create a corpus of all possible questions and keywords
        corpus = []
        for intent, keywords in self.service_keywords.items():
            corpus.extend(keywords)
            corpus.append(intent)
        
        # Add more natural language variations
        corpus.extend([
            "how can I check my current account balance",
            "where is the closest CGBank ATM location to me",
            "what are the current interest rates for savings accounts",
            "how do I transfer money to another person's account",
            "I've lost my debit card what should I do now",
            "what documents are required to open a new bank account",
            "what is the process to apply for a home loan",
            "what government schemes are available for farmers",
            "how can I generate my monthly account statement",
            "what is the customer support phone number for CGBank",
            "can you help me understand my spending patterns",
            "what are the benefits of a senior citizen account",
            "how do I activate my new debit card",
            "what is the daily withdrawal limit at ATMs",
            "how can I improve my financial health"
        ])
        
        # Train the vectorizer
        self.vectorizer.fit(corpus)
    
    def _create_knowledge_base(self) -> Dict[str, Any]:
        """Create a structured knowledge base from the JSON data with enhanced information"""
        kb = {
            'bank': CGBankDatabase.get_bank_info(),
            'loans': CGBankDatabase.get_loan_products(),
            'schemes': CGBankDatabase.get_government_schemes(),
            'accounts': CGBankDatabase.get_account_info(),
            'services': [
                "Savings Accounts",
                "Current Accounts",
                "Fixed Deposits",
                "Recurring Deposits",
                "Personal Loans",
                "Home Loans",
                "Education Loans",
                "Insurance Products",
                "Investment Services",
                "Digital Banking"
            ],
            'branches': CGBankDatabase.get_bank_info().get('branches', []),
            'atm_locations': CGBankDatabase.get_bank_info().get('atm_locations', []),
            'card_services': {
                'withdrawal_limit': 50000,
                'transaction_limit': 100000,
                'international_usage': True,
                'contactless_limit': 5000,
                'replacement_fee': 200,
                'card_types': ['Visa Platinum', 'Mastercard Gold', 'Rupay Select']
            },
            'interest_rates': {
                'savings': 3.5,
                'fixed_deposit': {
                    '7-14 days': 3.0,
                    '15-29 days': 3.25,
                    '30-90 days': 4.0,
                    '91-180 days': 5.0,
                    '181-364 days': 6.0,
                    '1-2 years': 6.5,
                    '2-3 years': 7.0,
                    '3-5 years': 7.25,
                    '5-10 years': 7.0
                },
                'loan': {
                    'home': 8.4,
                    'personal': 10.5,
                    'car': 7.9,
                    'education': 8.5,
                    'business': 12.0
                }
            },
            'investment_products': [
                {
                    'name': 'Fixed Deposit',
                    'min_amount': 5000,
                    'tenure': '7 days to 10 years',
                    'returns': '3.0% - 7.25%',
                    'features': ['Guanteed returns', 'Flexible tenure', 'Loan against FD']
                },
                {
                    'name': 'Recurring Deposit',
                    'min_amount': 1000,
                    'tenure': '6 months to 10 years',
                    'returns': '5.5% - 6.75%',
                    'features': ['Monthly savings', 'Disciplined investing', 'Fixed returns']
                },
                {
                    'name': 'Mutual Funds',
                    'min_amount': 500,
                    'tenure': 'Flexible',
                    'returns': 'Market linked',
                    'features': ['Diversified portfolio', 'Professional management', 'Tax benefits']
                }
            ],
            'security_info': {
                'tips': [
                    "Never share your OTP or password with anyone",
                    "CGBank will never ask for your sensitive information via email or phone",
                    "Always verify the URL before entering login credentials",
                    "Use strong, unique passwords for your banking accounts",
                    "Enable two-factor authentication for added security"
                ],
                'fraud_prevention': [
                    "Monitor your account regularly for suspicious activity",
                    "Register for instant transaction alerts",
                    "Immediately report lost or stolen cards",
                    "Use secure networks for banking transactions",
                    "Keep your contact information updated with the bank"
                ]
            },
            'financial_tips': [
                "Maintain an emergency fund of 3-6 months of expenses",
                "Pay yourself first - save at least 20% of your income",
                "Diversify your investments to manage risk",
                "Start retirement planning early for compound growth",
                "Review your financial plan at least once a year"
            ]
        }
        return kb
    
    def _get_random_response(self, response_type: str) -> str:
        """Get a random response of a given type with more variations"""
        responses = {
            'greetings': [
                "Hello! I'm Rexa, your CGBank assistant. How can I help you today?",
                "Hi there! Welcome to CGBank. What can I do for you?",
                "Greetings! I'm Rexa, ready to assist with your banking needs.",
                "Good day! How may I assist you with your CGBank account today?",
                "Welcome to CGBank digital services. How can I help you?"
            ],
            'thanks': [
                "You're welcome! Is there anything else I can help you with?",
                "Happy to help! Don't hesitate to ask if you have more questions.",
                "My pleasure! Let me know if you need anything else.",
                "Glad I could assist! Feel free to reach out anytime.",
                "Thank you for banking with us! Have a great day."
            ],
            'fallback': [
                "I'm not sure I understand. Could you rephrase that?",
                "I want to make sure I help you correctly. Could you provide more details?",
                "I'm still learning. Could you ask that in a different way?",
                "Let me connect you with a human representative for this question.",
                "I specialize in banking queries. Could you ask about account services?"
            ],
            'security': [
                "For security reasons, I can't access that information directly.",
                "To protect your account, please log in for that information.",
                "That's sensitive information - let me guide you securely.",
                "Your security is important. Let's handle that carefully.",
                "I'll need to verify your identity for that request."
            ]
        }
        
        return random.choice(responses.get(response_type, ["I'm here to help."]))
    
    def _extract_loan_info(self, loan_type: str) -> str:
        """Enhanced loan info extraction with detailed response formatting"""
        loans = CGBankDatabase.get_loan_products()
        
        # First try exact match with loan keys
        for key, loan_data in loans.items():
            if loan_type.lower() == key.lower():
                return self._format_loan_response(loan_data)
        
        # Then try matching with loan names
        for loan_data in loans.values():
            if loan_type.lower() in loan_data['name'].lower():
                return self._format_loan_response(loan_data)
        
        # Finally try fuzzy matching
        loan_names = [loan['name'].lower() for loan in loans.values()]
        matches = get_close_matches(loan_type.lower(), loan_names, n=1, cutoff=0.6)
        
        if matches:
            matched_loan_name = matches[0]
            for loan_data in loans.values():
                if loan_data['name'].lower() == matched_loan_name:
                    return self._format_loan_response(loan_data)
        
        return self._get_all_loans_info()
    
    def _format_loan_response(self, loan_data: Dict[str, Any]) -> str:
        """Format loan information into a detailed response"""
        eligibility = "\n".join([f"- {item}" for item in loan_data.get('eligibility', [])])
        documents = "\n".join([f"- {doc}" for doc in loan_data.get('documents', [])])
        features = "\n".join([f"- {feat}" for feat in loan_data.get('features', '').split(',')])
        
        return (f"**{loan_data['name']} at CGBank**\n\n"
               f"**Loan Amount:** {loan_data['amount']}\n"
               f"**Interest Rate:** {loan_data['interest']}\n"
               f"**Repayment Tenure:** {loan_data['tenure']}\n\n"
               f"**Key Features:**\n{features}\n\n"
               f"**Eligibility Criteria:**\n{eligibility}\n\n"
               f"**Required Documents:**\n{documents}\n\n"
               f"**How to Apply:** {loan_data.get('application', 'Visit any CGBank branch or apply online through our website or mobile app.')}\n\n"
               f"Would you like help calculating EMI for this loan?")
    
    def _get_all_loans_info(self) -> str:
        """Get comprehensive information about all loan products"""
        loans = CGBankDatabase.get_loan_products()
        response = "**Loan Products at CGBank:**\n\n"
        
        for loan in loans.values():
            response += (f"**{loan['name']}**\n"
                        f"- Amount: {loan['amount']}\n"
                        f"- Interest: {loan['interest']}\n"
                        f"- Tenure: {loan['tenure']}\n"
                        f"- Features: {loan.get('features', 'N/A')}\n\n")
        
        response += ("\nYou can ask about specific loans for more details, such as:\n"
                   "- 'Tell me about home loans'\n"
                   "- 'What are the requirements for a personal loan?'\n"
                   "- 'How do I apply for an education loan?'\n\n"
                   "I can also help you compare loan options or calculate EMIs.")
        return response
    
    def _extract_scheme_info(self, scheme_name: str) -> str:
        """Enhanced scheme info extraction with detailed formatting"""
        schemes = CGBankDatabase.get_government_schemes()
        
        # First try exact match with scheme keys
        for key, scheme_data in schemes.items():
            if scheme_name.lower() == key.lower():
                return self._format_scheme_response(scheme_data)
        
        # Then try matching with scheme names
        for scheme_data in schemes.values():
            if scheme_name.lower() in scheme_data['name'].lower():
                return self._format_scheme_response(scheme_data)
        
        # Finally try fuzzy matching
        scheme_names = [scheme['name'].lower() for scheme in schemes.values()]
        matches = get_close_matches(scheme_name.lower(), scheme_names, n=1, cutoff=0.6)
        
        if matches:
            matched_scheme_name = matches[0]
            for scheme_data in schemes.values():
                if scheme_data['name'].lower() == matched_scheme_name:
                    return self._format_scheme_response(scheme_data)
        
        return self._get_all_schemes_info()
    
    def _format_scheme_response(self, scheme_data: Dict[str, Any]) -> str:
        """Format scheme information into a detailed response"""
        benefits = "\n".join([f"- {benefit}" for benefit in scheme_data['benefits']])
        documents = "\n".join([f"- {doc}" for doc in scheme_data.get('documents', [])])
        
        return (f"**{scheme_data['name']}**\n\n"
               f"**Key Benefits:**\n{benefits}\n\n"
               f"**Eligibility:** {scheme_data['eligibility']}\n\n"
               f"**Required Documents:**\n{documents}\n\n"
               f"**How to Apply:** {scheme_data['application']}\n\n"
               f"Would you like help applying for this scheme?")
    
    def _get_all_schemes_info(self) -> str:
        """Get comprehensive information about all government schemes"""
        schemes = CGBankDatabase.get_government_schemes()
        response = "**Government Schemes Available Through CGBank:**\n\n"
        
        for scheme in schemes.values():
            response += f"**{scheme['name']}**\n"
            response += f"- Benefits: {', '.join(scheme['benefits'][:2])}...\n"
            response += f"- Eligibility: {scheme['eligibility']}\n\n"
        
        response += ("\nYou can ask about specific schemes for more details, such as:\n"
                    "- 'Tell me about PM Kisan scheme'\n"
                    "- 'What are the benefits of MUDRA loans?'\n"
                    "- 'How do I apply for StandUp India?'\n\n"
                    "I can guide you through the application process for any of these schemes.")
        return response
    
    def _extract_account_info(self, account_type: str) -> str:
        """Enhanced account info extraction with detailed formatting"""
        accounts = CGBankDatabase.get_account_info()
        
        # First try exact match with account keys
        for key, account_data in accounts.items():
            if account_type.lower() == key.lower():
                return self._format_account_response(account_data)
        
        # Then try matching with account names
        for account_data in accounts.values():
            if account_type.lower() in account_data.get('name', '').lower():
                return self._format_account_response(account_data)
        
        # Finally try fuzzy matching
        account_names = [account_data.get('name', '').lower() for account_data in accounts.values()]
        matches = get_close_matches(account_type.lower(), account_names, n=1, cutoff=0.6)
        
        if matches:
            matched_account_name = matches[0]
            for account_data in accounts.values():
                if account_data.get('name', '').lower() == matched_account_name:
                    return self._format_account_response(account_data)
        
        return self._get_all_accounts_info()
    
    def _get_account_creation_info(self) -> str:
        """Provide comprehensive information about account creation process"""
        accounts = CGBankDatabase.get_account_info()
        response = ("**Account Opening Process at CGBank**\n\n"
                   "Opening a new account with CGBank is simple and can be done through these steps:\n\n"
                   "1. **Choose Account Type**: Select the account that fits your needs\n"
                   "2. **Gather Documents**: Prepare the required KYC documents\n"
                   "3. **Visit Branch or Apply Online**: Complete the application process\n"
                   "4. **Initial Deposit**: Make the minimum required deposit\n"
                   "5. **Account Activation**: Your account will be activated within 24 hours\n\n"
                   "**Required Documents for All Accounts**:\n"
                   "- Proof of Identity (Aadhar card, PAN card, Passport, etc.)\n"
                   "- Proof of Address (Aadhar card, Utility bill, Rental agreement, etc.)\n"
                   "- Passport size photographs (2)\n"
                   "- Additional documents may be required based on account type\n\n"
                   "**Minimum Deposit Requirements**:\n")
        
        for account_type, account_data in accounts.items():
            response += f"- {account_data.get('name', account_type.replace('_', ' ').title())}: ₹{account_data.get('min_balance', 0):,.2f}\n"
        
        response += ("\n**Digital Account Opening**\n"
                    "You can start the process online by visiting our website or mobile app, "
                    "and then visit a branch to complete KYC verification.\n\n"
                    "Would you like me to help you choose the right account type based on your needs?")
        return response

    def _format_account_response(self, account_data: Dict[str, Any]) -> str:
        """Format account information into a comprehensive response"""
        try:
            name = account_data.get('name', 'Account')
            features = "\n".join([f"- {feature.strip()}" for feature in account_data.get('features', 'No special features').split(',')])
            benefits = "\n".join([f"- {benefit}" for benefit in account_data.get('benefits', [])])
            min_balance = account_data.get('min_balance', 0)
            interest_rate = account_data.get('interest_rate', 0)
            documents = "\n".join([f"- {doc.strip()}" for doc in account_data.get('documents', 'Not specified').split(',')])
            
            return (f"**{name} at CGBank**\n\n"
                   f"**Key Features:**\n{features}\n\n"
                   f"**Special Benefits:**\n{benefits if benefits else 'No special benefits'}\n\n"
                   f"**Minimum Balance Requirement:** ₹{min_balance:,.2f}\n"
                   f"**Interest Rate:** {interest_rate}% p.a.\n\n"
                   f"**Eligibility:** {account_data.get('eligibility', 'Not specified')}\n\n"
                   f"**Required Documents:**\n{documents}\n\n"
                   f"**How to Open:** {account_data.get('how_to_open', 'Visit any CGBank branch with the required documents')}\n\n"
                   f"Would you like to compare this with other account types?")
        except Exception as e:
            print(f"Error formatting account response: {e}")
            return "I'm having trouble retrieving the account details. Please try again later."
    
    def _get_all_accounts_info(self) -> str:
        """Get comprehensive information about all account types"""
        accounts = CGBankDatabase.get_account_info()
        response = "**Account Types at CGBank:**\n\n"
        
        for account_type, account_data in accounts.items():
            name = account_data.get('name', account_type.replace('_', ' ').title())
            min_balance = account_data.get('min_balance', 0)
            interest_rate = account_data.get('interest_rate', 0)
            
            response += (f"**{name}**\n"
                        f"- Min Balance: ₹{min_balance:,.2f}\n"
                        f"- Interest: {interest_rate}% p.a.\n"
                        f"- Best for: {account_data.get('best_for', 'General banking needs')}\n\n")
        
        response += ("\nYou can ask about specific accounts for more details, such as:\n"
                    "- 'Tell me about student accounts'\n"
                    "- 'What are the benefits of NRI accounts?'\n"
                    "- 'How do I open a senior citizen account?'\n\n"
                    "I can help you choose the right account based on your needs.")
        return response
    
    def _generate_monthly_report(self, username: str) -> Dict[str, Any]:
        """Generate a comprehensive monthly report with spending analysis"""
        transactions = CGBankDatabase.get_user_transactions(username)
        if not transactions:
            return None
        
        df = pd.DataFrame(transactions)
        last_month = datetime.now() - timedelta(days=30)
        df_last_month = df[df['date'] >= last_month]
        
        if df_last_month.empty:
            return None
        
        # Basic transaction metrics
        total_credit = df_last_month[df_last_month['amount'] > 0]['amount'].sum()
        total_debit = abs(df_last_month[df_last_month['amount'] < 0]['amount'].sum())
        net_change = total_credit - total_debit
        
        # Spending analysis by category
        spending_analysis = {}
        for category in df_last_month['category'].unique():
            spending_analysis[category] = df_last_month[df_last_month['category'] == category]['amount'].sum()
        
        # Average daily balance calculation
        daily_balances = []
        current_balance = CGBankDatabase.get_user(username)['balance']
        
        # Reconstruct daily balances by processing transactions in reverse
        for txn in sorted(df_last_month.to_dict('records'), key=lambda x: x['date'], reverse=True):
            daily_balances.append(current_balance)
            current_balance -= txn['amount']  # Subtract because we're going backward
        
        avg_balance = sum(daily_balances) / len(daily_balances) if daily_balances else current_balance
        
        # Interest earned calculation (simplified)
        interest_earned = avg_balance * (CGBankDatabase.get_user(username)['interest_rate'] / 100) / 12
        
        report = {
            'start_date': last_month.strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'total_transactions': len(df_last_month),
            'total_credit': total_credit,
            'total_debit': total_debit,
            'net_change': net_change,
            'avg_balance': avg_balance,
            'interest_earned': interest_earned,
            'spending_analysis': spending_analysis,
            'transactions': df_last_month.to_dict('records')
        }
        
        return report
    
    def _create_pdf_report(self, username: str, report_data: Dict[str, Any]) -> BytesIO:
        """Create a professional PDF report with enhanced formatting"""
        user_data = CGBankDatabase.get_user(username)
        if not user_data:
            return None
        
        try:
            pdf_buffer = PDFGenerator.generate_pdf_report(user_data, report_data)
            return pdf_buffer
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None
    
    def _create_download_link(self, pdf_buffer: BytesIO, username: str) -> str:
        """Create a secure download link for the PDF report"""
        try:
            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            filename = f"CGBank_Statement_{username}_{datetime.now().strftime('%Y%m%d')}.pdf"
            href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
            return href
        except Exception as e:
            print(f"Error creating download link: {e}")
            return "Error generating download link"
    
    def _get_ollama_response(self, message: str, context: str = "") -> str:
        """Get a response from Ollama LLM with enhanced banking context"""
        try:
            # Prepare the prompt with banking context and guidelines
            prompt = f"""
            You are Rexa, an AI banking assistant for CGBank (Coimbatore Trusted Banking Partner). 
            Your role is to provide accurate, helpful, and professional banking services to customers.
            
            **Current Context:**
            {context}
            
            **Knowledge Base Summary:**
            - Bank Name: CGBank
            - Services: Personal Banking, Business Banking, Loans, Investments
            - Account Types: {', '.join([a['name'] for a in self.knowledge_base['accounts'].values()])}
            - Loan Products: {', '.join([l['name'] for l in self.knowledge_base['loans'].values()])}
            
            **Customer Query:**
            {message}
            
            **Response Guidelines:**
            1. Be professional yet friendly (use "you" and "we" appropriately)
            2. Provide accurate information from the knowledge base
            3. If unsure, ask clarifying questions
            4. Keep responses concise (100-200 words max)
            5. Use Markdown formatting for better readability
            6. For account-specific queries, verify user is logged in
            7. Highlight important numbers/rates in bold
            8. End with a helpful follow-up question or suggestion
            9. Never share sensitive information without verification
            10. For complex queries, suggest contacting branch or customer support
            
            **Response:**
            """
            
            response = ollama.generate(
                model='banking-assistant',
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'max_tokens': 300,
                    'top_p': 0.9,
                    'frequency_penalty': 0.5,
                    'presence_penalty': 0.5
                }
            )
            
            # Post-process the response
            cleaned_response = response['choices'][0]['text'].strip()
            cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response)  # Remove excessive newlines
            cleaned_response = cleaned_response.replace("** ", "**").replace(" **", "**")  # Fix markdown formatting
            
            return cleaned_response
        except Exception as e:
            print(f"Error getting Ollama response: {e}")
            return "I'm having trouble processing your request. Please try again later or contact our customer support."
    
    def _identify_intent(self, message: str) -> Optional[str]:
        """Enhanced intent identification with NLP and context awareness"""
        message = message.lower()
        doc = nlp(message)
        
        # Check for greetings first
        if any(token.text.lower() in ['hello', 'hi', 'hey', 'greetings'] for token in doc[:3]):
            return 'greeting'
        
        # Check for thanks
        if any(token.text.lower() in ['thanks', 'thank', 'appreciate'] for token in doc):
            return 'thanks'
        
        # Preprocess the message with lemmatization and stopword removal
        processed_message = " ".join([token.lemma_ for token in doc 
                                     if not token.is_stop and not token.is_punct])
        
        # Transform the message using TF-IDF
        message_vec = self.vectorizer.transform([processed_message])
        
        # Calculate similarity with all known intents
        similarities = {}
        for intent, keywords in self.service_keywords.items():
            # Create a document for this intent by joining all keywords
            intent_doc = " ".join(keywords)
            intent_vec = self.vectorizer.transform([intent_doc])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(message_vec, intent_vec)[0][0]
            similarities[intent] = similarity
        
        # Get the most similar intent
        best_intent = max(similarities.items(), key=lambda x: x[1])
        
        # Only return if similarity is above threshold
        if best_intent[1] > 0.4:
            return best_intent[0]
        
        return None
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Enhanced entity extraction with financial context"""
        doc = nlp(message)
        entities = {
            'amounts': [],
            'dates': [],
            'account_types': [],
            'loan_types': [],
            'scheme_names': [],
            'time_periods': [],
            'locations': []
        }
        
        # Extract entities using spaCy NER and pattern matching
        for ent in doc.ents:
            if ent.label_ == 'MONEY':
                # Extract numerical value from money string
                amount = re.search(r'[\d,]+\.?\d*', ent.text)
                if amount:
                    try:
                        amount_value = float(amount.group().replace(',', ''))
                        entities['amounts'].append(amount_value)
                    except ValueError:
                        pass
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'ORG' or ent.label_ == 'PRODUCT':
                # Account types
                if 'account' in message.lower():
                    entities['account_types'].append(ent.text)
                # Loan types
                elif 'loan' in message.lower():
                    entities['loan_types'].append(ent.text)
                # Scheme names
                elif 'scheme' in message.lower() or 'program' in message.lower():
                    entities['scheme_names'].append(ent.text)
            elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
            elif ent.label_ == 'TIME':
                entities['time_periods'].append(ent.text)
        
        # Additional pattern matching for banking-specific entities
        # Account types
        account_pattern = r'\b(student|nri|senior|savings|current|business)\s?account\b'
        matches = re.finditer(account_pattern, message, re.IGNORECASE)
        for match in matches:
            entities['account_types'].append(match.group(1) + " account")
        
        # Loan types
        loan_pattern = r'\b(home|personal|car|auto|education|student|business)\s?loan\b'
        matches = re.finditer(loan_pattern, message, re.IGNORECASE)
        for match in matches:
            entities['loan_types'].append(match.group(1) + " loan")
        
        # Government schemes
        scheme_pattern = r'\b(pm kisan|pm svanidhi|standup india|mudra)\b'
        matches = re.finditer(scheme_pattern, message, re.IGNORECASE)
        for match in matches:
            entities['scheme_names'].append(match.group(0))
        
        return entities
    
    def _extract_amount_filters(self, message: str) -> Dict[str, float]:
        """Enhanced amount filter extraction with more patterns"""
        filters = {}
        
        # Patterns to match
        patterns = {
            'greater_than': r'(?:greater than|more than|above|over|higher than)\s*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)',
            'less_than': r'(?:less than|below|under|lower than)\s*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)',
            'between': r'between\s*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)\s*(?:and|to)\s*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)',
            'exact_amount': r'(?:exactly|precisely|amount of)\s*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)'
        }
        
        # Check for each pattern
        for filter_type, pattern in patterns.items():
            matches = re.search(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    if filter_type == 'between':
                        min_val = float(matches.group(1).replace(',', ''))
                        max_val = float(matches.group(2).replace(',', ''))
                        filters['min_amount'] = min_val
                        filters['max_amount'] = max_val
                    elif filter_type == 'exact_amount':
                        exact_val = float(matches.group(1).replace(',', ''))
                        filters['min_amount'] = exact_val - 1
                        filters['max_amount'] = exact_val + 1
                    else:
                        amount = float(matches.group(1).replace(',', ''))
                        if filter_type == 'greater_than':
                            filters['min_amount'] = amount
                        elif filter_type == 'less_than':
                            filters['max_amount'] = amount
                except (IndexError, ValueError):
                    pass
        
        return filters
    
    def _extract_date_filters(self, message: str) -> Dict[str, datetime]:
        """Enhanced date filter extraction with natural language support"""
        filters = {}
        doc = nlp(message)
        
        # Handle relative dates (last week, past month, etc.)
        relative_dates = {
            'today': datetime.now(),
            'yesterday': datetime.now() - timedelta(days=1),
            'last week': datetime.now() - timedelta(weeks=1),
            'past week': datetime.now() - timedelta(weeks=1),
            'last month': datetime.now() - timedelta(days=30),
            'past month': datetime.now() - timedelta(days=30),
            'last year': datetime.now() - timedelta(days=365),
            'this month': datetime.now().replace(day=1),
            'this year': datetime.now().replace(month=1, day=1)
        }
        
        for term, date in relative_dates.items():
            if term in message.lower():
                if 'from' in message.lower() or 'since' in message.lower():
                    filters['start_date'] = date
                elif 'to' in message.lower() or 'until' in message.lower():
                    filters['end_date'] = date
                else:
                    filters['start_date'] = date
                    filters['end_date'] = datetime.now()
        
        # Extract explicit dates
        date_ents = [ent for ent in doc.ents if ent.label_ == 'DATE']
        parsed_dates = []
        
        for ent in date_ents:
            try:
                # Try to parse as absolute date
                parsed_date = datetime.strptime(ent.text, '%d/%m/%Y')
                parsed_dates.append(parsed_date)
            except ValueError:
                try:
                    parsed_date = datetime.strptime(ent.text, '%d-%m-%Y')
                    parsed_dates.append(parsed_date)
                except ValueError:
                    # Handle relative dates not in our dictionary
                    if 'ago' in ent.text:
                        try:
                            num = int(re.search(r'\d+', ent.text).group())
                            if 'day' in ent.text:
                                parsed_dates.append(datetime.now() - timedelta(days=num))
                            elif 'week' in ent.text:
                                parsed_dates.append(datetime.now() - timedelta(weeks=num))
                            elif 'month' in ent.text:
                                parsed_dates.append(datetime.now() - timedelta(days=num*30))
                            elif 'year' in ent.text:
                                parsed_dates.append(datetime.now() - timedelta(days=num*365))
                        except:
                            pass
        
        # Sort dates and apply to filters
        parsed_dates = sorted(parsed_dates)
        
        if 'from' in message.lower() and 'to' in message.lower():
            if len(parsed_dates) >= 2:
                filters['start_date'] = parsed_dates[0]
                filters['end_date'] = parsed_dates[1]
        elif 'from' in message.lower() or 'since' in message.lower():
            if len(parsed_dates) >= 1:
                filters['start_date'] = parsed_dates[0]
        elif 'to' in message.lower() or 'until' in message.lower():
            if len(parsed_dates) >= 1:
                filters['end_date'] = parsed_dates[0]
        elif 'between' in message.lower():
            if len(parsed_dates) >= 2:
                filters['start_date'] = parsed_dates[0]
                filters['end_date'] = parsed_dates[1]
        elif parsed_dates:
            if len(parsed_dates) == 1:
                filters['start_date'] = parsed_dates[0]
                filters['end_date'] = datetime.now()
            else:
                filters['start_date'] = parsed_dates[0]
                filters['end_date'] = parsed_dates[-1]
        
        return filters
    
    def _filter_transactions(self, transactions: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced transaction filtering with multiple criteria"""
        filtered = transactions.copy()
        
        # Apply amount filters
        if 'min_amount' in filters:
            filtered = [t for t in filtered if abs(t['amount']) >= filters['min_amount']]
        if 'max_amount' in filters:
            filtered = [t for t in filtered if abs(t['amount']) <= filters['max_amount']]
        
        # Apply date filters
        if 'start_date' in filters:
            filtered = [t for t in filtered if t['date'] >= filters['start_date']]
        if 'end_date' in filters:
            filtered = [t for t in filtered if t['date'] <= filters['end_date']]
        
        # Apply category filter if present
        if 'category' in filters:
            filtered = [t for t in filtered if t.get('category', '').lower() == filters['category'].lower()]
        
        return filtered
    
    def _format_transactions_response(self, transactions: List[Dict[str, Any]], filters: Dict[str, Any]) -> str:
        """Format filtered transactions into a detailed response"""
        if not transactions:
            return "No transactions found matching your criteria."
        
        response = "Here are the transactions matching your request:\n\n"
        
        # Explain filters applied
        filter_explanation = []
        if 'min_amount' in filters:
            filter_explanation.append(f"amount greater than ₹{filters['min_amount']:,.2f}")
        if 'max_amount' in filters:
            filter_explanation.append(f"amount less than ₹{filters['max_amount']:,.2f}")
        if 'start_date' in filters:
            filter_explanation.append(f"from {filters['start_date'].strftime('%Y-%m-%d')}")
        if 'end_date' in filters:
            filter_explanation.append(f"to {filters['end_date'].strftime('%Y-%m-%d')}")
        if 'category' in filters:
            filter_explanation.append(f"category: {filters['category']}")
        
        if filter_explanation:
            response += f"**Filters applied:** {' and '.join(filter_explanation)}\n\n"
        
        # Add transaction details
        for i, txn in enumerate(transactions[:10], 1):
            sign = "+" if txn['amount'] > 0 else ""
            response += (f"{i}. **{txn['description']}** ({txn.get('category', 'Uncategorized')})\n"
                        f"   Amount: {sign}₹{abs(txn['amount']):,.2f}\n"
                        f"   Date: {txn['date'].strftime('%Y-%m-%d %H:%M')}\n"
                        f"   Balance: ₹{txn['balance']:,.2f}\n\n")
        
        if len(transactions) > 10:
            response += f"\nShowing 10 of {len(transactions)} matching transactions. "
            response += "Would you like to see more or refine your filters?"
        else:
            response += "\nWould you like to see more details about any of these transactions?"
        
        return response
    
    def _get_atm_info(self, location: Optional[str] = None) -> str:
        """Get comprehensive ATM information with location filtering"""
        atm_info = CGBankDatabase.get_atm_locations(location)
        
        response = "**ATM Services at CGBank:**\n\n"
        
        if atm_info:
            response += "**ATM Locations:**\n"
            for atm in atm_info[:5]:  # Limit to 5 nearest ATMs
                response += (f"- **{atm['location']}**\n"
                            f"  Address: {atm['address']}\n"
                            f"  Distance: {atm.get('distance', 'N/A')}\n")
            response += "\n"
        else:
            response += "No ATM locations found"
        
        card_services = self.knowledge_base.get('card_services', {})
        
        response += "**Card Services:**\n"
        response += f"- Daily Withdrawal Limit: ₹{card_services.get('withdrawal_limit', '50,000')}\n"
        response += f"- Transaction Limit: ₹{card_services.get('transaction_limit', '1,00,000')}\n"
        response += f"- International Usage: {'Yes' if card_services.get('international_usage', False) else 'No'}\n"
        response += f"- Contactless Limit: ₹{card_services.get('contactless_limit', '5,000')}\n\n"
        
        response += ("**For lost/stolen cards**, please call our 24/7 helpline immediately to block your card.\n"
                    "You can also temporarily block your card through our mobile app.")
        
        return response
    
    def _get_customer_support_info(self) -> str:
        """Get comprehensive customer support information"""
        bank_info = self.knowledge_base.get('bank', {})
        
        response = ("**CGBank Customer Support Channels**\n\n"
                   f"**Helpline:** {bank_info.get('helpline', '1800-123-4506')} (24/7)\n"
                   f"**Email:** {bank_info.get('email', 'support@cgbank.com')}\n"
                   f"**Chat Support:** Available on our website and mobile app\n"
                   f"**Branch Hours:** 9:30 AM - 4:30 PM (Mon-Fri), 9:30 AM - 1:30 PM (Sat)\n\n"
                   "**For complaints or grievances:**\n"
                   "1. Visit any branch and ask for the grievance officer\n"
                   "2. Email complaints@cgbank.com with your account details\n"
                   "3. Call our helpline and select the complaints option\n\n"
                   "We aim to resolve all complaints within 7 working days.")
        
        return response
    
    def _get_interest_rates_info(self) -> str:
        """Get current interest rates with detailed information"""
        rates = self.knowledge_base.get('interest_rates', {})
        
        response = "**Current Interest Rates at CGBank**\n\n"
        
        if rates.get('savings'):
            response += f"- **Savings Account:** {rates['savings']}% p.a.\n"
        
        if rates.get('fixed_deposit'):
            response += "- **Fixed Deposits:**\n"
            for tenure, rate in sorted(rates['fixed_deposit'].items(), 
                                     key=lambda x: x[1], reverse=True):
                response += f"  - {tenure}: {rate}% p.a.\n"
        
        if rates.get('loan'):
            response += "- **Loan Rates:**\n"
            for loan_type, rate in sorted(rates['loan'].items(), 
                                        key=lambda x: x[1]):
                response += f"  - {loan_type.replace('_', ' ').title()}: {rate}% p.a.\n"
        
        response += ("\n**Note:** Rates are subject to change. The rates mentioned are for "
                    "general reference only. Please contact us for the exact rates applicable "
                    "to your account or loan.")
        
        return response
    
    def _get_security_info(self) -> str:
        """Get comprehensive security and fraud prevention information"""
        security_info = self.knowledge_base.get('security_info', {})
        
        response = ("**Security Tips from CGBank**\n\n"
                   "Your security is our top priority. Here's how you can protect yourself:\n\n")
        
        if security_info.get('tips'):
            response += "**Essential Security Practices:**\n"
            for i, tip in enumerate(security_info['tips'], 1):
                response += f"{i}. {tip}\n"
            response += "\n"
        
        if security_info.get('fraud_prevention'):
            response += "**Fraud Prevention Measures:**\n"
            for i, measure in enumerate(security_info['fraud_prevention'], 1):
                response += f"{i}. {measure}\n"
            response += "\n"
        
        response += ("**If you suspect fraud:**\n"
                    "1. Contact us immediately through our helpline\n"
                    "2. Change your passwords and PINs\n"
                    "3. Monitor your account activity regularly\n"
                    "4. Report any unauthorized transactions within 3 days\n\n"
                    "Remember, CGBank will never ask for your password, OTP, or sensitive "
                    "information via email, phone, or SMS.")
        
        return response.rstrip()
    
    def _get_investment_info(self) -> str:
        """Get comprehensive investment product information"""
        investments = self.knowledge_base.get('investment_products', [])
        
        response = ("**Investment Options at CGBank**\n\n"
                   "Grow your wealth with our range of investment products:\n\n")
        
        for product in investments:
            response += (f"**{product['name']}**\n"
                        f"- Minimum Investment: ₹{product.get('min_amount', '5,000'):,}\n"
                        f"- Tenure: {product.get('tenure', 'Flexible')}\n"
                        f"- Expected Returns: {product.get('returns', 'Varies')}\n"
                        f"- Key Features: {product.get('features', 'N/A')}\n\n")
        
        response += ("Our financial advisors can help you choose the right investment based on your "
                    "goals and risk appetite. Book an appointment at any branch for personalized advice.")
        
        return response
    
    def _get_financial_advice(self) -> str:
        """Provide personalized financial advice"""
        tips = self.knowledge_base.get('financial_tips', [])
        
        response = ("**Financial Wellness Tips from CGBank**\n\n"
                   "Here are some recommendations to improve your financial health:\n\n")
        
        for i, tip in enumerate(tips, 1):
            response += f"{i}. {tip}\n"
        
        response += ("\nFor personalized financial planning advice, consider booking a "
                    "consultation with one of our financial advisors at your nearest branch.")
        
        return response
    
    def _is_personal_query(self, message: str) -> bool:
        """Enhanced personal query detection with NLP"""
        personal_keywords = ['my', 'mine', 'account', 'balance', 'transactions', 
                           'statement', 'details', 'i', 'me']
        possessive_phrases = ["what's my", "what is my", "show my", "tell me my", 
                             "check my", "view my", "see my", "access my"]
        
        message = message.lower()
        
        # Check for possessive phrases
        if any(phrase in message for phrase in possessive_phrases):
            return True
        
        # Check for personal keywords in context
        doc = nlp(message)
        for token in doc:
            if (token.text.lower() in personal_keywords and 
                token.dep_ in ('poss', 'attr', 'nsubj')):
                return True
        
        # Check for questions about the user
        if any(token.tag_ == 'WP' for token in doc):  # WH-pronoun (who, what, etc.)
            if 'i' in message or 'me' in message:
                return True
        
        return False
    
    def _handle_personal_query(self, message: str, username: str) -> str:
        """Handle personal account queries with enhanced responses"""
        user = CGBankDatabase.get_user(username)
        if not user:
            return "Please log in to access your account information."
        
        intent = self._identify_intent(message)
        entities = self._extract_entities(message)
        
        # Balance inquiry
        if intent == 'balance_inquiry':
            return (f"Your current account balance is **₹{user['balance']:,.2f}**.\n\n"
                   f"Account: {user['account_number']} ({user['account_type']})\n\n"
                   "Would you like to view recent transactions or make a transfer?")
        
        # Transaction history
        elif intent == 'transaction_history':
            # Check for transaction filters
            amount_filters = self._extract_amount_filters(message)
            date_filters = self._extract_date_filters(message)
            
            # Get transactions from session state or database
            transactions = CGBankDatabase.get_user_transactions(username)
            
            # Apply filters if any
            if amount_filters or date_filters:
                all_filters = {**amount_filters, **date_filters}
                filtered_transactions = self._filter_transactions(transactions, all_filters)
                return self._format_transactions_response(filtered_transactions, all_filters)
            else:
                # Default: show recent transactions
                recent_transactions = transactions[:5]
                if not recent_transactions:
                    return "You don't have any transactions yet."
                
                response = "Here are your recent transactions:\n\n"
                for i, txn in enumerate(recent_transactions, 1):
                    sign = "+" if txn['amount'] > 0 else ""
                    response += (f"{i}. **{txn['description']}** ({txn.get('category', 'Uncategorized')})\n"
                                f"   Amount: {sign}₹{abs(txn['amount']):,.2f}\n"
                                f"   Date: {txn['date'].strftime('%Y-%m-%d %H:%M')}\n"
                                f"   Balance: ₹{txn['balance']:,.2f}\n\n")
                response += "Would you like to filter these transactions by amount, date, or category?"
                return response
        
        # Monthly report
        elif intent == 'monthly_report':
            report = self._generate_monthly_report(username)
            if not report:
                return "You don't have enough transactions to generate a monthly report yet."
            
            # Generate PDF report
            pdf_buffer = self._create_pdf_report(username, report)
            if pdf_buffer:
                # Store the download link in session state
                download_link = self._create_download_link(pdf_buffer, username)
                st.session_state.download_link = download_link
                
                response = (f"**📊 Monthly Report ({report['start_date']} to {report['end_date']})**\n\n"
                          f"**Total Transactions:** {report['total_transactions']}\n"
                          f"**Total Credit:** ₹{report['total_credit']:,.2f}\n"
                          f"**Total Debit:** ₹{report['total_debit']:,.2f}\n"
                          f"**Net Change:** ₹{report['net_change']:,.2f}\n\n"
                          "Your PDF report is ready! Would you like to download it now? (Yes/No)")
            else:
                response = "I couldn't generate the PDF report. Please try again later."
            
            return response
        
        # Account information
        elif intent == 'account_info':
            return (f"**Your Account Details:**\n\n"
                   f"**Account Holder:** {user['name']}\n"
                   f"**Account Number:** {user['account_number']}\n"
                   f"**Account Type:** {user['account_type']}\n"
                   f"**Current Balance:** ₹{user['balance']:,.2f}\n\n"
                   "Would you like to know more about your account features or services?")
        
        # Default response for personal queries
        return ("I can help you with your account details, transactions, and more.\n\n"
               "You can ask me about:\n"
               "- Your current balance\n"
               "- Recent transactions\n"
               "- Monthly statements\n"
               "- Account services\n\n"
               "What would you like to know?")
    
    def process_message(self, message: str, username: Optional[str] = None) -> str:
        """Enhanced message processing with context awareness and personalization"""
        message = message.strip()
        if not message:
            return "Please type your question or request."
        
        # Check for greetings
        if any(word in message.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            if username:
                user_data = CGBankDatabase.get_user(username)
                if user_data:
                    return f"Hello {user_data['name']}! {self._get_random_response('greetings')}"
            return self._get_random_response('greetings')
        
        # Check for thanks
        if any(word in message.lower() for word in ['thank', 'thanks', 'appreciate']):
            return self._get_random_response('thanks')
        
        # Check if this is a personal account query
        is_personal = self._is_personal_query(message)
        
        # Handle personal queries if user is logged in
        if is_personal and username:
            return self._handle_personal_query(message, username)
        elif is_personal:
            return ("Please log in to access your personal account information.\n\n"
                   "I can still help with general banking questions about accounts, "
                   "loans, or other services.")
        
        # Identify intent for non-personal queries
        intent = self._identify_intent(message)
        entities = self._extract_entities(message)
        
        # Handle balance inquiries (non-personal)
        if intent == 'balance_inquiry':
            return ("To check your account balance, please log in to your account.\n\n"
                   "For general information about account types and features, you can ask:\n"
                   "- 'What types of accounts does CGBank offer?'\n"
                   "- 'What is the minimum balance for a savings account?'")
        
        # Handle account information requests
        elif intent == 'account_info':
            if entities.get('account_types'):
                account_type = entities['account_types'][0]
                return self._extract_account_info(account_type)
            elif any(word in message.lower() for word in ['student', 'nri', 'senior', 'regular', 'current']):
                if 'student' in message.lower():
                    return self._extract_account_info('student_account')
                elif 'nri' in message.lower():
                    return self._extract_account_info('nri_account')
                elif 'senior' in message.lower():
                    return self._extract_account_info('senior_account')
                elif 'current' in message.lower():
                    return self._extract_account_info('current_account')
                else:
                    return self._extract_account_info('regular_savings_account')
            elif any(word in message.lower() for word in ['create', 'open', 'new']):
                return self._get_account_creation_info()
            else:
                return self._get_all_accounts_info()
        
        # Handle loan information requests
        elif intent == 'loan_info':
            if entities.get('loan_types'):
                return self._extract_loan_info(entities['loan_types'][0])
            elif any(word in message.lower() for word in ['home', 'personal', 'car', 'education', 'business']):
                if 'home' in message.lower():
                    return self._extract_loan_info('home_loan')
                elif 'personal' in message.lower():
                    return self._extract_loan_info('personal_loan')
                elif 'car' in message.lower() or 'auto' in message.lower():
                    return self._extract_loan_info('car_loan')
                elif 'education' in message.lower():
                    return self._extract_loan_info('education_loan')
                elif 'business' in message.lower():
                    return self._extract_loan_info('business_loan')
            else:
                return self._get_all_loans_info()
        
        # Handle scheme information requests
        elif intent == 'scheme_info':
            if entities.get('scheme_names'):
                return self._extract_scheme_info(entities['scheme_names'][0])
            elif any(word in message.lower() for word in ['kisan', 'svanidhi', 'standup', 'mudra']):
                if 'kisan' in message.lower():
                    return self._extract_scheme_info('pm_kisan_scheme')
                elif 'svanidhi' in message.lower():
                    return self._extract_scheme_info('pm_svanidhi_scheme')
                elif 'standup' in message.lower():
                    return self._extract_scheme_info('standup_india_scheme')
                elif 'mudra' in message.lower():
                    return self._extract_scheme_info('mudra_loan_scheme')
            else:
                return self._get_all_schemes_info()
        
        # Handle ATM and card services
        elif intent == 'atm_info':
            if entities.get('locations'):
                return self._get_atm_info(entities['locations'][0])
            return self._get_atm_info()
        
        # Handle card services
        elif intent == 'card_info':
            return self._get_atm_info()  # Reusing the same function as it contains card info
        
        # Handle customer support
        elif intent == 'customer_support':
            return self._get_customer_support_info()
        
        # Handle interest rates
        elif intent == 'interest_rates':
            return self._get_interest_rates_info()
        
        # Handle security info
        elif intent == 'security_info':
            return self._get_security_info()
        
        # Handle investment info
        elif intent == 'investment_info':
            return self._get_investment_info()
        
        # Handle financial advice
        elif intent == 'financial_advice':
            return self._get_financial_advice()
        
        # Handle bank information
        elif intent == 'bank_info':
            bank_info = CGBankDatabase.get_bank_info()
            if 'branch' in message.lower() or 'location' in message.lower():
                branches = "\n".join([f"- **{branch['name']}**: {branch['address']} ({branch.get('timings', '')})" 
                                    for branch in bank_info['branches'][:3]])
                return f"**CGBank Branches:**\n{branches}"
            elif 'service' in message.lower() or 'product' in message.lower():
                services = "\n".join([f"- {service}" for service in bank_info['services']])
                return f"**CGBank Services:**\n{services}"
            elif 'time' in message.lower() or 'hour' in message.lower():
                timings = bank_info['branches'][0]['timings']
                return f"**Branch Timings:**\n{timings}"
            else:
                return (f"**About {bank_info['name']}:**\n"
                       f"{bank_info['tagline']}\n\n"
                       f"**Address:** {bank_info['address']}\n"
                       f"**Contact:** {bank_info['contact']}\n"
                       f"**Email:** {bank_info['email']}\n"
                       f"**Helpline:** {bank_info['helpline']}")
        
        # For all other queries, use Ollama with context
        context = ""
        if username:
            user_data = CGBankDatabase.get_user(username)
            if user_data:
                context = (f"Customer: {user_data['name']}\n"
                          f"Account Type: {user_data['account_type']}\n"
                          f"Last Login: {datetime.now().strftime('%Y-%m-%d')}")
        
        return self._get_ollama_response(message, context)

class CGBankApp:
    """Enhanced Streamlit application for CGBank with improved UI/UX"""
    
    def __init__(self):
        self.bot = RexaBot()
        self.feedback_system = FeedbackSystem()
        self._initialize_session_state()
        self._setup_page_config()
        self._load_custom_styles()
    
    def _initialize_session_state(self):
        """Initialize session state variables with enhanced structure"""
        session_defaults = {
            'logged_in': False,
            'current_user': None,
            'page': "login",
            'bot_conversation': [],
            'show_popup_bot': False,
            'transactions': [],
            'download_link': None,
            'feedback_submitted': False,
            'show_create_account': False,
            'login_attempts': 0,
            'last_attempt': datetime.now()
        }
        
        for key, value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _setup_page_config(self):
        """Configure the Streamlit page settings with enhanced options"""
        st.set_page_config(
            page_title="CGBank - Digital Banking",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://www.cgbank.com/help',
                'Report a problem': 'https://www.cgbank.com/report',
                'About': "CGBank Digital Banking v2.0"
            }
        )
    
    def _load_custom_styles(self):
        """Load enhanced CSS styles for better UI/UX"""
        st.markdown("""
        <style>
            /* Main header styling */
            .main-header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                padding: 2rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            
            /* Account card styling */
            .account-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .account-card:hover {
                transform: translateY(-5px);
            }
            
            /* Chat message styling */
            .bot-message {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                color: #333;
                border-left: 4px solid #2a5298;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .user-message {
                background: #e3f2fd;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                color: #333;
                border-left: 4px solid #1976d2;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            /* Popup bot styling */
            .popup-bot {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 350px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.2);
                z-index: 1000;
                border: 1px solid #e0e0e0;
                overflow: hidden;
            }
            .popup-bot-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .popup-bot-content {
                max-height: 300px;
                overflow-y: auto;
                padding: 1rem;
            }
            
            /* Transaction item styling */
            .transaction-item {
                background: white;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }
            .transaction-item:hover {
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            /* Popup message styling */
            .popup-user-message {
                background: #e3f2fd;
                padding: 8px 12px;
                border-radius: 12px;
                margin: 8px 0;
                margin-left: auto;
                max-width: 80%;
                font-size: 0.9em;
            }
            .popup-bot-message {
                background: #f8f9fa;
                padding: 8px 12px;
                border-radius: 12px;
                margin: 8px 0;
                max-width: 80%;
                font-size: 0.9em;
            }
            
            /* Popup toggle button */
            .popup-toggle-btn {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 1001;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: white;
                border: none;
                cursor: pointer;
                padding: 0;
                margin: 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            .popup-toggle-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .popup-toggle-btn img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                border-radius: 50%;
            }
            
            /* Quick action buttons */
            .quick-action-btn {
                margin: 0.2rem 0;
                width: 100%;
                transition: all 0.3s ease;
            }
            .quick-action-btn:hover {
                transform: translateY(-2px);
            }
            
            /* Report card styling */
            .report-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            /* Feedback form styling */
            .feedback-form {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            /* Star rating styling */
            .star-rating {
                display: flex;
                justify-content: center;
                margin: 1rem 0;
            }
            .star-rating input {
                display: none;
            }
            .star-rating label {
                font-size: 2rem;
                color: #ddd;
                cursor: pointer;
                margin: 0 0.2rem;
                transition: color 0.2s;
            }
            .star-rating input:checked ~ label {
                color: #ffc107;
            }
            .star-rating label:hover,
            .star-rating label:hover ~ label {
                color: #ffc107;
            }
            
            /* Create account form */
            .create-account-form {
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                margin-top: 2rem;
            }
            
            /* Markdown text styling */
            .markdown-text {
                font-family: Arial, sans-serif;
                line-height: 1.6;
            }
            .markdown-text strong {
                color: #2a5298;
            }
            .markdown-text ul {
                padding-left: 1.5rem;
            }
            .markdown-text li {
                margin-bottom: 0.5rem;
            }
            
            /* Sidebar navigation */
            .sidebar-nav-item {
                padding: 0.75rem 1rem;
                margin: 0.25rem 0;
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            .sidebar-nav-item:hover {
                background-color: #f0f2f6;
            }
            .sidebar-nav-item.active {
                background-color: #e3f2fd;
                border-left: 4px solid #2a5298;
            }
            
            /* Form input styling */
            .stTextInput input, .stNumberInput input, .stTextArea textarea {
                border-radius: 5px !important;
                border: 1px solid #ddd !important;
            }
            
            /* Button styling */
            .stButton>button {
                border-radius: 5px !important;
                border: none !important;
                background-color: #2a5298 !important;
                color: white !important;
                transition: all 0.3s ease !important;
            }
            .stButton>button:hover {
                background-color: #1e3c72 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            ::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_feedback_form(self):
        """Render the enhanced feedback form in the dashboard"""
        with st.expander("📝 Share Your Feedback", expanded=False):
            with st.form("feedback_form", clear_on_submit=True):
                st.markdown("### We Value Your Feedback")
                st.markdown("Help us improve our services by sharing your experience.")
                
                # Name and email inputs
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Your Name*", placeholder="Enter your name")
                with col2:
                    email = st.text_input("Your Email", placeholder="Enter your email")
                
                # Star rating with emojis
                st.markdown("### How would you rate your experience?*")
                rating = st.slider("Rating", 1, 5, 5, 
                                 format="%d ⭐", 
                                 key="feedback_rating",
                                 help="1 = Poor, 5 = Excellent")
                
                # Feedback text area
                feedback = st.text_area("Your Feedback*", 
                                      placeholder="What did you like or what can we improve?",
                                      height=150,
                                      help="Please share your detailed feedback")
                
                # Submit button with validation
                submitted = st.form_submit_button("Submit Feedback", 
                                                use_container_width=True,
                                                help="Your feedback helps us improve")
                
                if submitted:
                    if not name or not feedback:
                        st.error("Please provide both your name and feedback!")
                        return
                    
                    # Send feedback email
                    success = self.feedback_system.send_feedback_email(
                        name=name,
                        email=email if email else "Not provided",
                        rating=rating,
                        feedback=feedback
                    )
                    
                    if success:
                        st.success("""
                        Thank you for your feedback! 💙
                        
                        We appreciate your time and will use your feedback to improve our services.
                        """)
                        st.session_state.feedback_submitted = True
                        st.balloons()
                    else:
                        st.error("""
                        Failed to submit feedback. 😔
                        
                        Please try again later or contact our customer support.
                        """)
    
    def _render_login_page(self):
        """Render the enhanced login page with account creation option"""
        st.markdown("""
        <div class="main-header">
            <h1 style="font-size: 2.5rem;">🏦 CGBank</h1>
            <h3 style="margin-bottom: 0.5rem;">Coimbatore Trusted Banking Partner</h3>
            <h4 style="margin-top: 0; font-weight: 400;">Secure • Reliable • Innovative</h4>
            <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 0.9em;">Head Office</div>
                    <div>174/2 Avinashi Road, Coimbatore</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em;">Contact</div>
                    <div>+91-63820-74060</div>
                </div>
            </div>
            <p style="margin-top: 1rem; font-size: 0.9em;">HELPLINE: 1800-123-4506 | Email: Cgbankcbe@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.session_state.show_create_account:
                self._render_create_account_form()
            else:
                self._render_login_form()
    
    def _render_login_form(self):
        """Render the enhanced login form with security features"""
        st.markdown("### Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username", 
                                   placeholder="Enter your username",
                                   help="Your CGBank username")
            password = st.text_input("Password", 
                                   type="password",
                                   placeholder="Enter your password",
                                   help="Your CGBank password")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Login", 
                                                use_container_width=True,
                                                help="Access your account")
            with col2:
                create_account = st.form_submit_button("Create New Account", 
                                                     use_container_width=True,
                                                     help="Open a new CGBank account")
            
            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password!")
                    return
                
                # Simple rate limiting
                if (datetime.now() - st.session_state.last_attempt).seconds < 5:
                    st.session_state.login_attempts += 1
                else:
                    st.session_state.login_attempts = 1
                    st.session_state.last_attempt = datetime.now()
                
                if st.session_state.login_attempts > 3:
                    st.error("Too many login attempts. Please try again after 5 minutes.")
                    return
                
                try:
                    # Verify user credentials
                    if CGBankDatabase.verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.current_user = username.lower()
                        st.session_state.page = "dashboard"
                        st.session_state.transactions = CGBankDatabase.get_user_transactions(username)
                        st.session_state.login_attempts = 0
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
                except Exception as e:
                    st.error(f"An error occurred during login: {str(e)}")
            
            if create_account:
                st.session_state.show_create_account = True
                st.rerun()
    
    def _render_create_account_form(self):
        """Render the enhanced account creation form with validation"""
        st.markdown("### Create a New CGBank Account")
        st.markdown("Join CGBank today and enjoy our premium banking services.")
        
        with st.form("create_account_form"):
            st.markdown("#### Personal Information")
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name*", 
                                         placeholder="Enter your full name",
                                         help="As per your identity proof")
            with col2:
                email = st.text_input("Email Address*", 
                                    placeholder="Enter your email address",
                                    help="For account communication")
            
            col1, col2 = st.columns(2)
            with col1:
                phone = st.text_input("Phone Number*", 
                                    placeholder="Enter your 10-digit mobile number",
                                    help="For OTP verification")
            with col2:
                dob = st.date_input("Date of Birth*",
                                  min_value=datetime(1900, 1, 1),
                                  max_value=datetime.now(),
                                  help="Must be 18+ years")
            
            address = st.text_area("Residential Address*", 
                                 placeholder="Enter your full address with PIN code",
                                 help="As per your address proof")
            
            st.markdown("#### Account Details")
            account_type = st.selectbox(
                "Account Type*",
                options=["Student Account", "NRI Account", "Senior Citizen Account", 
                        "Regular Savings Account", "Current Account"],
                index=3,
                help="Select the account type that fits your needs"
            )
            
            st.markdown("#### Login Credentials")
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input("Choose Username*", 
                                       placeholder="Create a username",
                                       help="4-20 characters (letters, numbers)")
            with col2:
                password = st.text_input("Create Password*", 
                                       type="password",
                                       placeholder="Create a password",
                                       help="Minimum 8 characters with mix of letters and numbers")
            
            confirm_password = st.text_input("Confirm Password*", 
                                           type="password",
                                           placeholder="Confirm your password",
                                           help="Must match the password above")
            
            st.markdown("#### KYC Information")
            col1, col2 = st.columns(2)
            with col1:
                aadhar_number = st.text_input("Aadhar Number*", 
                                            placeholder="Enter 12-digit Aadhar number",
                                            help="As per your Aadhar card")
            with col2:
                pan_number = st.text_input("PAN Number", 
                                         placeholder="Enter PAN number",
                                         help="Required for certain accounts").upper()
            
            # Terms and conditions
            agree = st.checkbox("I agree to the Terms and Conditions*", 
                              help="Please read our terms before proceeding")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Submit Application", 
                                             use_container_width=True,
                                             help="Submit your account application")
            with col2:
                cancel = st.form_submit_button("Cancel", 
                                             use_container_width=True,
                                             help="Return to login page")
            
            if cancel:
                st.session_state.show_create_account = False
                st.rerun()
            
            if submit:
                # Validate all required fields
                if not all([full_name, email, phone, dob, address, username, password, 
                           confirm_password, aadhar_number, agree]):
                    st.error("Please fill in all required fields!")
                    return
                
                # Validate password match
                if password != confirm_password:
                    st.error("Passwords do not match!")
                    return
                
                # Validate password strength
                if len(password) < 8:
                    st.error("Password must be at least 8 characters!")
                    return
                if not re.search(r'[A-Z]', password):
                    st.error("Password must contain at least one uppercase letter!")
                    return
                if not re.search(r'[a-z]', password):
                    st.error("Password must contain at least one lowercase letter!")
                    return
                if not re.search(r'[0-9]', password):
                    st.error("Password must contain at least one number!")
                    return
                
                # Validate Aadhar number
                if len(aadhar_number) != 12 or not aadhar_number.isdigit():
                    st.error("Please enter a valid 12-digit Aadhar number!")
                    return
                
                # Validate PAN number if provided
                if pan_number and (len(pan_number) != 10 or 
                                  not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan_number)):
                    st.error("Please enter a valid PAN number!")
                    return
                
                # Validate phone number
                if len(phone) != 10 or not phone.isdigit():
                    st.error("Please enter a valid 10-digit phone number!")
                    return
                
                # Validate age
                age = (datetime.now() - datetime.combine(dob, datetime.min.time())).days // 365
                if age < 18:
                    st.error("You must be at least 18 years old to open an account!")
                    return
                if account_type == "Senior Citizen Account" and age < 60:
                    st.error("You must be at least 60 years old for a Senior Citizen account!")
                    return
                
                # Check if username already exists
                if CGBankDatabase.get_user(username):
                    st.error("Username already exists! Please choose another one.")
                    return
                
                # Prepare user data
                user_data = {
                    "name": full_name,
                    "email": email,
                    "phone": phone,
                    "address": address,
                    "dob": dob.strftime('%Y-%m-%d'),
                    "account_type": account_type,
                    "aadhar_number": aadhar_number,
                    "pan_number": pan_number if pan_number else "Not provided",
                    "balance": 1000.0 if account_type != "Student Account" else 0.0
                }
                
                # Create the user account
                if CGBankDatabase.create_user(username, password, user_data):
                    st.session_state.show_create_account = False
                    st.success("""
                    🎉 Your account has been created successfully!
                    
                    You can now login with your username and password.
                    
                    **Next Steps:**
                    - Visit any CGBank branch to complete KYC verification
                    - Download our mobile app for digital banking
                    - Set up transaction alerts for security
                    
                    Thank you for choosing CGBank!
                    """)
                    st.balloons()
                else:
                    st.error("""
                    Failed to create account. 😔
                    
                    Please try again or visit our nearest branch for assistance.
                    """)
    
    def _render_dashboard(self):
        """Render the enhanced dashboard with financial overview"""
        user = CGBankDatabase.get_user(st.session_state.current_user)
        if not user:
            st.error("User not found!")
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.page = "login"
            st.rerun()
            return
        
        st.markdown(f"""
        <div class="main-header">
            <h1 style="font-size: 2rem;">Welcome back, {user['name']}! 👋</h1>
            <p style="margin-bottom: 0;">Account: {user['account_number']} | {user['account_type']}</p>
            <p style="margin-top: 0.5rem; font-size: 1.2rem;">💰 Current Balance: ₹{user['balance']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="account-card">
                <h3>💳 Monthly Spending</h3>
                <h2>₹12,340.50</h2>
                <p>↓ 5.2% from last month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="account-card">
                <h3>📈 Monthly Income</h3>
                <h2>₹25,000.00</h2>
                <p>↑ 3.5% from last month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="account-card">
                <h3>🏦 Savings Rate</h3>
                <h2>32.5%</h2>
                <p>↑ 2.1% from last month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="account-card">
                <h3>📅 Upcoming Bills</h3>
                <h2>₹2,340.50</h2>
                <p>3 bills due this week</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ⚡ Quick Actions")
        cols = st.columns(5)
        with cols[0]:
            if st.button("💸 Transfer Money", key="dashboard_transfer", use_container_width=True):
                st.session_state.page = "transfer"
                st.rerun()
        with cols[1]:
            if st.button("💰 Pay Bills", key="dashboard_bills", use_container_width=True):
                st.session_state.page = "bills"
                st.rerun()
        with cols[2]:
            if st.button("📊 Transactions", key="dashboard_transactions", use_container_width=True):
                st.session_state.page = "transactions"
                st.rerun()
        with cols[3]:
            if st.button("📈 Reports", key="dashboard_reports", use_container_width=True):
                st.session_state.page = "reports"
                st.rerun()
        with cols[4]:
            if st.button("🤖 Chat with Rexa", key="dashboard_rexa", use_container_width=True):
                st.session_state.page = "rexa"
                st.rerun()
        
        # Financial overview
        st.markdown("### 📊 Financial Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Monthly Spending by Category")
            categories = CGBankDatabase.get_spending_categories(st.session_state.current_user)
            df = pd.DataFrame(categories)
            
            if not df.empty:
                fig = px.pie(df, values='amount', names='name', 
                            title="Spending Distribution",
                            hole=0.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No spending data available for analysis")
        
        with col2:
            st.markdown("#### Recent Transactions")
            transactions = CGBankDatabase.get_user_transactions(st.session_state.current_user, limit=5)
            
            if transactions:
                for txn in transactions:
                    color = "#28a745" if txn['amount'] > 0 else "#dc3545"
                    sign = "+" if txn['amount'] > 0 else "-"
                    
                    st.markdown(f"""
                    <div class="transaction-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0;">{txn['description']}</h4>
                                <p style="margin: 0; color: #6c757d; font-size: 0.8em;">
                                    {txn['date'].strftime('%b %d, %Y')} • {txn.get('category', 'Other')}
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <h4 style="margin: 0; color: {color};">{sign}₹{abs(txn['amount']):,.2f}</h4>
                                <p style="margin: 0; color: #6c757d; font-size: 0.8em;">
                                    Balance: ₹{txn['balance']:,.2f}
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("View All Transactions", key="view_all_txns", use_container_width=True):
                    st.session_state.page = "transactions"
                    st.rerun()
            else:
                st.info("No recent transactions found")
        
        # Upcoming bills
        st.markdown("### 📅 Upcoming Bills")
        bills = CGBankDatabase.get_user_bills(st.session_state.current_user)
        
        if bills:
            for bill in bills[:3]:
                status_color = "#ffc107" if bill["status"] == "Due Soon" else "#dc3545"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div style="margin-bottom: 0.5rem;">
                        <h4 style="margin: 0;">{bill['name']}</h4>
                        <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
                            Due: {bill['due']} • <span style="color: {status_color};">{bill['status']}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button(f"Pay ₹{bill['amount']:,.2f}", key=f"pay_{bill['name']}", use_container_width=True):
                        st.session_state.page = "bills"
                        st.rerun()
            
            if len(bills) > 3:
                if st.button("View All Bills", key="view_all_bills", use_container_width=True):
                    st.session_state.page = "bills"
                    st.rerun()
        else:
            st.info("No upcoming bills found")
        
        # Add feedback form to dashboard
        if not st.session_state.feedback_submitted:
            self._render_feedback_form()
    
    def _render_report_page(self):
        """Render the enhanced report analysis page"""
        st.markdown("### 📊 Financial Reports & Analysis")
        
        if not st.session_state.transactions:
            st.session_state.transactions = CGBankDatabase.get_user_transactions(st.session_state.current_user)
        
        df = pd.DataFrame(st.session_state.transactions)
        
        if df.empty or 'date' not in df.columns:
            st.warning("No transaction data available for analysis")
            return
        
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        default_start = max(min_date, max_date - timedelta(days=30))
        default_end = max_date
        
        st.markdown("#### Select Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                     value=default_start,
                                     min_value=min_date,
                                     max_value=max_date,
                                     key="report_start_date")
        with col2:
            end_date = st.date_input("End Date", 
                                   value=default_end,
                                   min_value=min_date,
                                   max_value=max_date,
                                   key="report_end_date")
        
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
        
        if filtered_df.empty:
            st.warning("No transactions found for the selected date range")
            return
        
        # Summary statistics
        st.markdown("#### 📋 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="report-card">
                <h4>Total Transactions</h4>
                <h2>{len(filtered_df)}</h2>
                <p style="color: #6c757d; font-size: 0.9em;">
                    {sum(filtered_df['amount'] > 0)} credits • {sum(filtered_df['amount'] < 0)} debits
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_credit = filtered_df[filtered_df['amount'] > 0]['amount'].sum()
            st.markdown(f"""
            <div class="report-card">
                <h4>Total Credit</h4>
                <h2>₹{total_credit:,.2f}</h2>
                <p style="color: #6c757d; font-size: 0.9em;">
                    {len(filtered_df[filtered_df['amount'] > 0])} transactions
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_debit = abs(filtered_df[filtered_df['amount'] < 0]['amount'].sum())
            st.markdown(f"""
            <div class="report-card">
                <h4>Total Debit</h4>
                <h2>₹{total_debit:,.2f}</h2>
                <p style="color: #6c757d; font-size: 0.9em;">
                    {len(filtered_df[filtered_df['amount'] < 0])} transactions
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Transaction trends
        st.markdown("#### 📈 Transaction Trends")
        daily_trends = filtered_df.set_index('date').resample('D')['amount'].sum().reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(daily_trends, x='date', y='amount', 
                         title="Daily Transaction Amounts",
                         labels={'amount': 'Amount (₹)', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(filtered_df[filtered_df['amount'] < 0].groupby(
                filtered_df['date'].dt.day_name())['amount'].sum().abs().reset_index(),
                x='date', y='amount',
                title="Weekly Spending Pattern",
                labels={'amount': 'Amount (₹)', 'date': 'Day of Week'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Spending categories
        st.markdown("#### 🗂️ Spending Categories")
        category_df = filtered_df[filtered_df['amount'] < 0].groupby('category')['amount'].agg(['sum', 'count']).reset_index()
        category_df['sum'] = category_df['sum'].abs()
        category_df = category_df.sort_values('sum', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(category_df, values='sum', names='category', 
                        title="Amount by Category",
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(category_df, x='category', y='sum',
                        title="Spending by Category",
                        labels={'sum': 'Amount (₹)', 'category': 'Category'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction details
        st.markdown("#### 📜 Transaction Details")
        display_df = filtered_df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['amount'] = display_df['amount'].apply(lambda x: f"+₹{x:,.2f}" if x > 0 else f"-₹{abs(x):,.2f}")
        st.dataframe(display_df[['date', 'description', 'category', 'amount']], 
                    hide_index=True,
                    use_container_width=True)
        
        # Export options
        st.markdown("#### 📤 Export Report")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate PDF Report", key="generate_pdf", use_container_width=True):
                report = {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'total_transactions': len(filtered_df),
                    'total_credit': total_credit,
                    'total_debit': total_debit,
                    'net_change': total_credit - total_debit,
                    'transactions': filtered_df.to_dict('records')
                }
                
                pdf_buffer = self._create_pdf_report(st.session_state.current_user, report)
                if pdf_buffer:
                    download_link = self._create_download_link(pdf_buffer, st.session_state.current_user)
                    st.session_state.download_link = download_link
                    st.rerun()
        
        if 'download_link' in st.session_state and st.session_state.download_link:
            st.markdown(st.session_state.download_link, unsafe_allow_html=True)
    
    def _render_bot_page(self):
        """Render the enhanced chatbot interface"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Rexa - Your Personal Banking Assistant</h1>
            <p>Ask me anything about your account, transactions, or banking services</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display conversation history
        for conv in st.session_state.bot_conversation[-10:]:
            if isinstance(conv, dict) and 'user' in conv and 'bot' in conv:
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {conv['user']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="bot-message markdown-text">
                    <strong>🤖 Rexa:</strong> {conv['bot']}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your message to Rexa:", 
                                     placeholder="Ask me about your account, transactions, or banking services...",
                                     key="chat_input",
                                     label_visibility="collapsed")
            
            col1, col2, col3 = st.columns(3)
            with col2:
                submitted = st.form_submit_button("Send", use_container_width=True)
            
            if submitted and user_input:
                try:
                    bot_response = self.bot.process_message(
                        user_input, 
                        st.session_state.current_user if st.session_state.logged_in else None
                    )
                    
                    st.session_state.bot_conversation.append({
                        'user': user_input,
                        'bot': bot_response
                    })
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing message: {str(e)}")

    def _render_popup_bot(self):
        """Render the enhanced popup bot interface"""
        # Robot GIF URL
        robot_gif_url = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjVtY2V4bHJ5Z3R0eWJ4c3B6dHh0bHZ5d2V5d3J2dXZ5Z2F4eWZ4ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HU7JI8AfUAbM5HO/giphy.gif"
        
        # Create a button with the GIF as its content
        st.markdown(f"""
        <button class="popup-toggle-btn" onclick="document.getElementById('toggle-bot').click()">
            <img src="{robot_gif_url}" alt="Chat with Rexa">
        </button>
        """, unsafe_allow_html=True)
        
        # Hidden checkbox to control the popup state
        show_popup = st.checkbox("Toggle Bot", key="toggle_bot", label_visibility="hidden")
        
        if show_popup:
            st.session_state.show_popup_bot = True
        else:
            st.session_state.show_popup_bot = False
        
        if st.session_state.show_popup_bot:
            st.markdown("""
            <div class="popup-bot">
                <div class="popup-bot-header">
                    <h4 style="margin: 0;">🤖 Rexa - Banking Assistant</h4>
                    <button onclick="document.getElementById('toggle-bot').click()" 
                            style="background: none; border: none; color: white; cursor: pointer;">
                        ×
                    </button>
                </div>
                <div class="popup-bot-content">
            """, unsafe_allow_html=True)
            
            # Display recent messages
            for conv in st.session_state.bot_conversation[-3:]:
                if isinstance(conv, dict) and 'user' in conv and 'bot' in conv:
                    st.markdown(f"""
                    <div class="popup-user-message">
                        <strong>You:</strong> {conv['user']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="popup-bot-message markdown-text">
                        <strong>Rexa:</strong> {conv['bot']}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
                </div>
                <div style="padding: 1rem; border-top: 1px solid #eee;">
                    <p style="margin-bottom: 0.5rem; font-weight: bold;">Quick Banking Commands:</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            """, unsafe_allow_html=True)
            
            # Quick action buttons
            quick_actions = [
                ("💰 Balance", "What's my current balance?"),
                ("📊 Transactions", "Show my recent transactions"),
                ("💸 Transfer", "I want to transfer money"),
                ("🧾 Bills", "Show my upcoming bills"),
                ("🏦 Branches", "Where is the nearest branch?"),
                ("📅 Statement", "Generate my account statement")
            ]
            
            for action, cmd in quick_actions:
                if st.button(action, key=f"popup_{action}", 
                           help=cmd, 
                           use_container_width=True):
                    self._handle_popup_action(cmd)
            
            st.markdown("""
                    </div>
            """, unsafe_allow_html=True)
            
            # Chat input
            with st.form("popup_chat_form", clear_on_submit=True):
                user_input = st.text_input("Type your message:", 
                                         key="popup_input",
                                         label_visibility="collapsed")
                
                submitted = st.form_submit_button("Send", 
                                                use_container_width=True)
                
                if submitted and user_input:
                    self._handle_popup_action(user_input)
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _handle_popup_action(self, message: str):
        """Handle an action from the popup bot"""
        try:
            bot_response = self.bot.process_message(
                message, 
                st.session_state.current_user if st.session_state.logged_in else None
            )
            
            st.session_state.bot_conversation.append({
                'user': message,
                'bot': bot_response
            })
            
            st.rerun()
        except Exception as e:
            st.error(f"Error handling popup action: {str(e)}")
    
    def _render_sidebar(self):
        """Render the enhanced sidebar navigation"""
        with st.sidebar:
            st.markdown("### 🏦 CGBank Navigation")
            
            if st.session_state.logged_in:
                user = CGBankDatabase.get_user(st.session_state.current_user)
                if not user:
                    st.error("User data not found!")
                    return
                
                st.markdown(f"**Welcome, {user['name']}**")
                st.markdown(f"Account: {user['account_number']}")
                st.markdown(f"Balance: ₹{user['balance']:,.2f}")
                
                # Navigation items
                nav_items = [
                    ("🏠 Dashboard", "dashboard"),
                    ("📊 Transactions", "transactions"),
                    ("💸 Transfer", "transfer"),
                    ("💰 Bills", "bills"),
                    ("📈 Reports", "reports"),
                    ("🤖 Rexa", "rexa")
                ]
                
                for item in nav_items:
                    if st.button(item[0], 
                               key=f"sidebar_{item[1]}", 
                               use_container_width=True,
                               help=f"Go to {item[0]}"):
                        st.session_state.page = item[1]
                        st.rerun()
                
                st.markdown("---")
                
                # User actions
                if st.button("🔒 Logout", 
                           key="sidebar_logout", 
                           use_container_width=True,
                           help="Logout of your account"):
                    st.session_state.logged_in = False
                    st.session_state.current_user = None
                    st.session_state.page = "login"
                    st.rerun()
                
                st.markdown("---")
                st.markdown("**Need Help?**")
                st.markdown("[Contact Support](https://www.cgbank.com/support)")
                st.markdown("[FAQs](https://www.cgbank.com/faq)")
            else:
                st.markdown("Please login to access your account")
                st.markdown("---")
                st.markdown("**New to CGBank?**")
                st.markdown("[Open an Account](https://www.cgbank.com/open-account)")
                st.markdown("[Explore Services](https://www.cgbank.com/services)")
    
    def _render_transactions_page(self):
        """Render the enhanced transactions page"""
        st.markdown("### 📋 Transaction History")
        
        if not st.session_state.transactions:
            st.session_state.transactions = CGBankDatabase.get_user_transactions(st.session_state.current_user)
        
        # Filter options
        st.markdown("#### Filter Transactions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_filter = st.selectbox("Date Range", 
                                     ["Last 7 days", "Last 30 days", "Last 90 days", "All", "Custom"],
                                     index=1)
        
        with col2:
            category_filter = st.selectbox("Category", 
                                         ["All"] + sorted(list(set(t.get('category', 'Other') 
                                                                  for t in st.session_state.transactions))))
        
        with col3:
            amount_filter = st.selectbox("Amount Range",
                                       ["All", "Less than ₹1,000", "₹1,000 - ₹5,000", "₹5,000 - ₹10,000", "Above ₹10,000"])
        
        # Apply filters
        filtered_transactions = st.session_state.transactions.copy()
        
        # Date filter
        if date_filter == "Last 7 days":
            cutoff_date = datetime.now() - timedelta(days=7)
            filtered_transactions = [t for t in filtered_transactions if t['date'] >= cutoff_date]
        elif date_filter == "Last 30 days":
            cutoff_date = datetime.now() - timedelta(days=30)
            filtered_transactions = [t for t in filtered_transactions if t['date'] >= cutoff_date]
        elif date_filter == "Last 90 days":
            cutoff_date = datetime.now() - timedelta(days=90)
            filtered_transactions = [t for t in filtered_transactions if t['date'] >= cutoff_date]
        elif date_filter == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From", 
                                         value=datetime.now() - timedelta(days=30),
                                         max_value=datetime.now())
            with col2:
                end_date = st.date_input("To", 
                                       value=datetime.now(),
                                       min_value=start_date,
                                       max_value=datetime.now())
            
            filtered_transactions = [t for t in filtered_transactions 
                                   if start_date <= t['date'].date() <= end_date]
        
        # Category filter
        if category_filter != "All":
            filtered_transactions = [t for t in filtered_transactions 
                                   if t.get('category', 'Other') == category_filter]
        
        # Amount filter
        if amount_filter == "Less than ₹1,000":
            filtered_transactions = [t for t in filtered_transactions 
                                   if abs(t['amount']) < 1000]
        elif amount_filter == "₹1,000 - ₹5,000":
            filtered_transactions = [t for t in filtered_transactions 
                                   if 1000 <= abs(t['amount']) <= 5000]
        elif amount_filter == "₹5,000 - ₹10,000":
            filtered_transactions = [t for t in filtered_transactions 
                                   if 5000 <= abs(t['amount']) <= 10000]
        elif amount_filter == "Above ₹10,000":
            filtered_transactions = [t for t in filtered_transactions 
                                   if abs(t['amount']) > 10000]
        
        # Display transactions
        if not filtered_transactions:
            st.info("No transactions found matching your filters")
            return
        
        # Summary stats
        total_credit = sum(t['amount'] for t in filtered_transactions if t['amount'] > 0)
        total_debit = abs(sum(t['amount'] for t in filtered_transactions if t['amount'] < 0))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Credit", f"₹{total_credit:,.2f}")
        with col2:
            st.metric("Total Debit", f"₹{total_debit:,.2f}")
        
        # Transaction list
        for txn in filtered_transactions[:50]:  # Limit to 50 transactions
            color = "#28a745" if txn['amount'] > 0 else "#dc3545"
            sign = "+" if txn['amount'] > 0 else "-"
            
            with st.expander(f"{txn['date'].strftime('%b %d, %Y')}: {txn['description']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Category:** {txn.get('category', 'Other')}")
                    st.markdown(f"**Date:** {txn['date'].strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    st.markdown(f"**Amount:** <span style='color: {color};'>{sign}₹{abs(txn['amount']):,.2f}</span>", 
                               unsafe_allow_html=True)
                    st.markdown(f"**Balance:** ₹{txn['balance']:,.2f}")
        
        if len(filtered_transactions) > 50:
            st.info(f"Showing 50 of {len(filtered_transactions)} transactions. Adjust filters to see more.")
    
    def _render_transfer_page(self):
        """Render the enhanced fund transfer page"""
        st.markdown("### 💸 Transfer Money")
        
        user = CGBankDatabase.get_user(st.session_state.current_user)
        if not user:
            st.error("User data not found!")
            return
        
        # Transfer form
        with st.form("transfer_form"):
            st.markdown("#### Transfer Details")
            
            col1, col2 = st.columns(2)
            with col1:
                recipient_name = st.text_input("Recipient Name*", 
                                             placeholder="Enter recipient's name",
                                             help="As per the recipient's bank records")
            with col2:
                recipient_account = st.text_input("Recipient Account Number*", 
                                                placeholder="Enter account number",
                                                help="10-digit account number")
            
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Amount (₹)*", 
                                       min_value=0.01,
                                       step=0.01,
                                       format="%.2f",
                                       help="Amount to transfer")
            with col2:
                transfer_type = st.selectbox("Transfer Type",
                                           ["IMPS", "NEFT", "RTGS"],
                                           help="IMPS: Instant, NEFT: Within hours, RTGS: Large amounts")
            
            description = st.text_input("Description (Optional)", 
                                      placeholder="What's this transfer for?",
                                      max_chars=50,
                                      help="Appears on both accounts")
            
            # Beneficiary selection
            st.markdown("#### Saved Beneficiaries")
            beneficiaries = [
                {"name": "John Doe", "account": "1234567890", "bank": "CGBank"},
                {"name": "Jane Smith", "account": "9876543210", "bank": "Other Bank"}
            ]
            
            selected_beneficiary = None
            cols = st.columns(3)
            for i, beneficiary in enumerate(beneficiaries):
                with cols[i % 3]:
                    if st.button(f"{beneficiary['name']}\n{beneficiary['account']}", 
                               key=f"beneficiary_{i}",
                               use_container_width=True):
                        selected_beneficiary = beneficiary
            
            if selected_beneficiary:
                st.info(f"Selected: {selected_beneficiary['name']} ({selected_beneficiary['account']})")
                recipient_name = selected_beneficiary['name']
                recipient_account = selected_beneficiary['account']
            
            # Submit button
            submitted = st.form_submit_button("Initiate Transfer", 
                                            use_container_width=True,
                                            help="Review details before submitting")
            
            if submitted:
                if not all([recipient_name, recipient_account, amount]):
                    st.error("Please fill in all required fields!")
                    return
                
                if amount > user['balance']:
                    st.error("Insufficient funds for this transfer!")
                    return
                
                if len(recipient_account) != 10 or not recipient_account.isdigit():
                    st.error("Please enter a valid 10-digit account number!")
                    return
                
                # Confirm transfer
                with st.expander("Confirm Transfer Details", expanded=True):
                    st.markdown(f"""
                    **From:** {user['name']} (A/c {user['account_number']})
                    
                    **To:** {recipient_name} (A/c {recipient_account})
                    
                    **Amount:** ₹{amount:,.2f}
                    
                    **Transfer Type:** {transfer_type}
                    
                    **Description:** {description if description else 'Not specified'}
                    """)
                    
                    confirm = st.checkbox("I confirm the details are correct", 
                                        help="Please verify all details before proceeding")
                    
                    if st.button("Confirm Transfer", 
                               disabled=not confirm,
                               use_container_width=True):
                        try:
                            # Use the database method to ensure JSON is updated
                            success = CGBankDatabase.add_transaction(
                                st.session_state.current_user,
                                f'Transfer to {recipient_account}',
                                -amount
                            )
                            
                            if success:
                                st.success(f"""
                                Transfer of ₹{amount:,.2f} to {recipient_name} initiated successfully!
                                
                                **Reference Number:** {random.randint(1000000000, 9999999999)}
                                
                                The amount should reflect in the recipient's account within:
                                - IMPS: Few minutes
                                - NEFT: Within 2 hours
                                - RTGS: Immediately (for amounts over ₹2 lakhs)
                                """)
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("Failed to process transfer. Please try again.")
                        except Exception as e:
                            st.error(f"Error processing transfer: {str(e)}")
    
    def _render_bills_page(self):
        """Render the enhanced bills payment page"""
        st.markdown("### 💰 Bill Payments")
        
        user = CGBankDatabase.get_user(st.session_state.current_user)
        if not user:
            st.error("User data not found!")
            return
        
        # Biller categories
        biller_categories = {
            "Utilities": ["Electricity", "Water", "Gas", "Internet", "Mobile"],
            "Insurance": ["Health Insurance", "Vehicle Insurance", "Life Insurance"],
            "Subscriptions": ["OTT Platforms", "Music Streaming", "Cloud Storage"]
        }
        
        # Tab interface
        tab1, tab2 = st.tabs(["Pay Bills", "Add New Biller"])
        
        with tab1:
            st.markdown("#### Upcoming Bills")
            bills = CGBankDatabase.get_user_bills(st.session_state.current_user)
            
            if bills:
                for bill in bills:
                    status_color = "#ffc107" if bill["status"] == "Due Soon" else "#dc3545"
                    
                    with st.expander(f"{bill['name']} - Due: {bill['due']}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            **Amount Due:** ₹{bill['amount']:,.2f}
                            
                            **Status:** <span style="color: {status_color};">{bill['status']}</span>
                            
                            **Last Paid:** {bill.get('last_paid', 'Never')}
                            """, unsafe_allow_html=True)
                        with col2:
                            if st.button("Pay Now", 
                                       key=f"pay_{bill['name']}",
                                       use_container_width=True):
                                try:
                                    if bill['amount'] > user['balance']:
                                        st.error("Insufficient funds to pay this bill!")
                                    else:
                                        # Use the database method to ensure JSON is updated
                                        success = CGBankDatabase.add_bill_payment(
                                            st.session_state.current_user,
                                            bill['name'],
                                            bill['amount']
                                        )
                                        
                                        if success:
                                            st.success(f"""
                                            Payment of ₹{bill['amount']:,.2f} for {bill['name']} processed successfully!
                                            
                                            **Transaction ID:** {random.randint(1000000000, 9999999999)}
                                            """)
                                            st.balloons()
                                            st.rerun()
                                        else:
                                            st.error("Payment processed but failed to update records.")
                                except Exception as e:
                                    st.error(f"Error processing bill payment: {str(e)}")
            else:
                st.info("No upcoming bills found")
        
        with tab2:
            st.markdown("#### Add New Biller")
            with st.form("add_biller_form"):
                st.markdown("Select biller category and enter details to add a new biller.")
                
                col1, col2 = st.columns(2)
                with col1:
                    category = st.selectbox("Biller Category*",
                                          list(biller_categories.keys()),
                                          help="Select the category for this biller")
                with col2:
                    biller_name = st.selectbox("Biller Name*",
                                             biller_categories[category],
                                             help="Select the biller name")
                
                account_number = st.text_input("Account Number*",
                                             placeholder="Your account number with the biller",
                                             help="As per your bill")
                
                col1, col2 = st.columns(2)
                with col1:
                    amount = st.number_input("Amount (₹)*",
                                           min_value=0.01,
                                           step=0.01,
                                           format="%.2f",
                                           help="Current amount due")
                with col2:
                    due_date = st.date_input("Due Date*",
                                           min_value=datetime.now().date(),
                                           help="When payment is due")
                
                # Submit button
                submitted = st.form_submit_button("Add Biller", 
                                                use_container_width=True,
                                                help="Save this biller for future payments")
                
                if submitted:
                    if not all([category, biller_name, account_number, amount, due_date]):
                        st.error("Please fill in all required fields!")
                        return
                    
                    # Create new bill data
                    new_bill = {
                        "name": f"{category} - {biller_name}",
                        "account": account_number,
                        "amount": amount,
                        "due": due_date.strftime("%Y-%m-%d"),
                        "status": "Due Soon",
                        "last_paid": "Never"
                    }
                    
                    # Add the bill
                    success = CGBankDatabase.add_new_bill(st.session_state.current_user, new_bill)
                    if success:
                        st.success(f"""
                        Biller {new_bill['name']} added successfully!
                        
                        You can now pay this bill from your dashboard or this page.
                        """)
                        st.rerun()
                    else:
                        st.error("Failed to add new biller. Please try again.")
    
    def run(self):
        """Run the enhanced application"""
        self._render_sidebar()
        
        if st.session_state.logged_in:
            self._render_popup_bot()
            
            if st.session_state.page == "dashboard":
                self._render_dashboard()
            elif st.session_state.page == "transactions":
                self._render_transactions_page()
            elif st.session_state.page == "transfer":
                self._render_transfer_page()
            elif st.session_state.page == "bills":
                self._render_bills_page()
            elif st.session_state.page == "reports":
                self._render_report_page()
            elif st.session_state.page == "rexa":
                self._render_bot_page()
        else:
            self._render_login_page()

if __name__ == "__main__":
    app = CGBankApp()
    app.run()