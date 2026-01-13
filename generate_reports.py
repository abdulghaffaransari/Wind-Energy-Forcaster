"""
Comprehensive PDF Report Generator for Wind Energy Forecasting Project
Generates detailed reports for Data Overview, Analysis, Model Training, Predictions, and Dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Try to import reportlab, if not available, use fpdf
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    try:
        from fpdf import FPDF
        FPDF_AVAILABLE = True
    except ImportError:
        FPDF_AVAILABLE = False
        print("Warning: Neither reportlab nor fpdf is available. Installing reportlab...")
        os.system("pip install reportlab")
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            REPORTLAB_AVAILABLE = True
        except:
            print("Could not install reportlab. Please install manually: pip install reportlab")
            sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_processing.data_loader import DataLoader
from data_processing.feature_engineering import FeatureEngineer
from utils.metrics import calculate_metrics
import yaml


class PDFReportGenerator:
    """Generate comprehensive PDF reports"""
    
    def __init__(self, output_dir="Reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Load data
        print("Loading data...")
        loader = DataLoader()
        self.df = loader.load_data()
        self.df_processed = FeatureEngineer().create_all_features(self.df.copy())
        
        # Load config
        with open("src/config/config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load metrics
        metrics_path = "outputs/reports/model_metrics.csv"
        if Path(metrics_path).exists():
            self.metrics_df = pd.read_csv(metrics_path, index_col=0)
        else:
            self.metrics_df = None
    
    def _add_logo(self, story):
        """Add university logo to report if available (black version for reports)"""
        logo_path = Path("assets/logos/university_logo_black.png")
        if not logo_path.exists():
            logo_path = Path("assets/logos/university_logo.png")  # Fallback to white
        if logo_path.exists():
            try:
                logo_img = Image(str(logo_path), width=3*inch, height=1.2*inch)
                story.append(logo_img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                pass  # Continue without logo if there's an error
        return story
    
    def _add_student_footer(self, story):
        """Add student name at the end of report"""
        story.append(Spacer(1, 0.5*inch))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("─" * 50, self.styles['Normal']))
        student_style = ParagraphStyle('StudentFooter', parent=self.styles['Normal'], 
            fontSize=11, textColor=colors.HexColor('#1565c0'), 
            alignment=TA_CENTER, spaceAfter=5, spaceBefore=10)
        story.append(Paragraph("Student: <b>Abdul Ghaffar Ansari</b>", student_style))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}", 
            ParagraphStyle('DateFooter', parent=self.styles['Normal'], 
            fontSize=9, textColor=colors.HexColor('#666666'), 
            alignment=TA_CENTER, spaceAfter=10)))
        return story
    
    def _setup_custom_styles(self):
        """Setup enhanced colorful paragraph styles"""
        # Enhanced Title - Large, bold, blue
        self.title_style = ParagraphStyle(
            'EnhancedTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=20,
            spaceBefore=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Subtitle
        self.subtitle_style = ParagraphStyle(
            'EnhancedSubtitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1565c0'),
            spaceAfter=15,
            spaceBefore=15,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Main heading - Dark blue with background
        self.heading_style = ParagraphStyle(
            'EnhancedHeading',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#0d47a1'),
            spaceAfter=12,
            spaceBefore=15,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#e3f2fd'),
            borderWidth=1,
            borderColor=colors.HexColor('#1976d2'),
            borderPadding=5
        )
        
        # Subheading - Medium blue
        self.subheading_style = ParagraphStyle(
            'EnhancedSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#1976d2'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        # Body text with justification
        self.body_style = ParagraphStyle(
            'EnhancedBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#212121'),
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leading=14
        )
    
    def _create_colorful_table(self, data, col_widths, header_color, row_colors, title=""):
        """Create a colorful table with alternating row colors"""
        table = Table(data, colWidths=col_widths)
        
        # Create table style
        table_style = [
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), header_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1.5, colors.black),
        ]
        
        # Add alternating row colors
        for i in range(1, len(data)):
            color = row_colors[i % len(row_colors)]
            table_style.append(('BACKGROUND', (0, i), (-1, i), color))
            table_style.append(('TEXTCOLOR', (0, i), (-1, i), colors.HexColor('#000000')))
            table_style.append(('FONTSIZE', (0, i), (-1, i), 10))
            table_style.append(('TOPPADDING', (0, i), (-1, i), 8))
            table_style.append(('BOTTOMPADDING', (0, i), (-1, i), 8))
        
        table.setStyle(TableStyle(table_style))
        return table
    
    def generate_data_overview_report(self):
        """Generate Enhanced Data Overview Report with extensive detail"""
        print("Generating Enhanced Data Overview Report...")
        filename = self.output_dir / "01_Data_Overview_Report.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=A4, 
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
        story = []
        
        # Add University Logo
        self._add_logo(story)
        
        # COVER PAGE
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("WIND ENERGY FORECASTING PROJECT", self.title_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Comprehensive Data Overview Report", self.subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("A Detailed Analysis of Wind Energy Dataset",
                              ParagraphStyle('Subtitle', fontSize=14, alignment=TA_CENTER,
                                           textColor=colors.HexColor('#616161'))))
        story.append(Spacer(1, 1*inch))
        story.append(PageBreak())
        
        # TABLE OF CONTENTS
        story.append(Paragraph("Table of Contents", self.heading_style))
        toc_items = [
            "1. Executive Summary",
            "2. Introduction to Wind Energy Forecasting",
            "3. Dataset Overview and Characteristics",
            "4. Data Collection and Sources",
            "5. Data Structure and Format",
            "6. Statistical Summary and Descriptive Analysis",
            "7. Data Quality Assessment",
            "8. Temporal Characteristics",
            "9. Variable Descriptions and Meanings",
            "10. Key Insights and Observations",
            "11. Data Preprocessing Overview",
            "12. Conclusion and Next Steps"
        ]
        for item in toc_items:
            story.append(Paragraph(item, self.body_style))
            story.append(Spacer(1, 0.1*inch))
        story.append(PageBreak())
        
        # 1. EXECUTIVE SUMMARY
        story.append(Paragraph("1. Executive Summary", self.heading_style))
        story.append(Paragraph(
            "This comprehensive report provides an in-depth analysis of the wind energy dataset used for "
            "developing machine learning forecasting models. The dataset contains daily measurements of "
            "wind power generation, installed capacity, and temperature for Germany spanning from 2017 to 2019. "
            "This analysis serves as the foundational understanding required for building accurate and reliable "
            "forecasting models.",
            self.body_style
        ))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<b>Key Highlights:</b>", self.subheading_style))
        highlights = [
            f"• Total dataset contains <b>{len(self.df):,} daily records</b>, providing comprehensive coverage",
            f"• Time period spans <b>{(self.df.index.max() - self.df.index.min()).days + 1} days</b> from {self.df.index.min().date()} to {self.df.index.max().date()}",
            f"• Average daily wind generation: <b>{self.df['wind_generation_actual'].mean():,.0f} MW</b>",
            f"• Peak wind generation recorded: <b>{self.df['wind_generation_actual'].max():,.0f} MW</b>",
            f"• Average capacity utilization: <b>{(self.df['wind_generation_actual'] / self.df['wind_capacity']).mean() * 100:.2f}%</b>",
        ]
        for highlight in highlights:
            story.append(Paragraph(highlight, self.body_style))
        story.append(PageBreak())
        
        # 2. INTRODUCTION
        story.append(Paragraph("2. Introduction to Wind Energy Forecasting", self.heading_style))
        story.append(Paragraph(
            "Wind energy has emerged as one of the most important renewable energy sources in the global "
            "transition towards sustainable power generation. Accurate forecasting of wind power generation "
            "is crucial for energy grid management, economic planning, and ensuring reliable power supply.",
            self.body_style
        ))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("<b>What is Wind Energy Forecasting?</b>", self.subheading_style))
        story.append(Paragraph(
            "Wind energy forecasting involves predicting the amount of electrical power that will be generated "
            "by wind turbines over a specific time period. This prediction is based on historical patterns, "
            "weather conditions, and various meteorological factors. Accurate forecasts help energy operators "
            "plan power generation, manage grid stability, and optimize energy trading.",
            self.body_style
        ))
        story.append(PageBreak())
        
        # 3. DATASET OVERVIEW
        story.append(Paragraph("3. Dataset Overview and Characteristics", self.heading_style))
        story.append(Paragraph(
            "The dataset used in this project contains comprehensive daily measurements of wind energy "
            "generation in Germany. Understanding the dataset structure and characteristics is essential "
            "for developing effective forecasting models.",
            self.body_style
        ))
        story.append(Spacer(1, 0.2*inch))
        
        data_info = [
            ['Attribute', 'Value', 'Explanation'],
            ['Total Records', f"{len(self.df):,}", 'Number of daily observations in the dataset'],
            ['Date Range Start', f"{self.df.index.min().date()}", 'First date of data collection'],
            ['Date Range End', f"{self.df.index.max().date()}", 'Last date of data collection'],
            ['Total Days', f"{(self.df.index.max() - self.df.index.min()).days + 1}", 'Complete time span covered'],
            ['Number of Variables', f"{len(self.df.columns)}", 'Total features/columns in the dataset'],
            ['Data Frequency', 'Daily', 'Measurements taken once per day'],
            ['Geographic Region', 'Germany', 'Country where data was collected'],
        ]
        
        info_table = self._create_colorful_table(
            data_info,
            col_widths=[2*inch, 2*inch, 2.5*inch],
            header_color=colors.HexColor('#1565c0'),
            row_colors=[colors.HexColor('#e3f2fd'), colors.HexColor('#bbdefb')]
        )
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("<b>What Do These Numbers Mean?</b>", self.subheading_style))
        story.append(Paragraph(
            f"The dataset contains <b>{len(self.df):,} daily records</b>, which means we have wind energy measurements for "
            f"approximately <b>{(self.df.index.max() - self.df.index.min()).days / 365.25:.1f} years</b> of data. This is a substantial "
            "amount of historical data that allows machine learning models to learn patterns, seasonal trends, and relationships "
            "between variables. Having multiple years of data is crucial because it enables models to capture yearly "
            "seasonal patterns, which are important for accurate forecasting.",
            self.body_style
        ))
        story.append(PageBreak())
        
        # 6. STATISTICAL SUMMARY
        story.append(Paragraph("6. Statistical Summary and Descriptive Analysis", self.heading_style))
        story.append(Paragraph(
            "Statistical analysis helps us understand the distribution, central tendencies, and variability "
            "of our data. These statistics are fundamental for understanding what 'normal' values look like "
            "and identifying potential outliers or unusual patterns.",
            self.body_style
        ))
        story.append(Spacer(1, 0.15*inch))
        
        stats = self.df[['wind_generation_actual', 'wind_capacity', 'temperature']].describe()
        stats_data = [['Statistic'] + stats.columns.tolist()]
        for idx in stats.index:
            row = [idx]
            for col in stats.columns:
                val = stats.loc[idx, col]
                if idx in ['mean', 'std']:
                    row.append(f"{val:,.2f}")
                else:
                    row.append(f"{val:,.0f}" if col != 'temperature' else f"{val:.2f}")
            stats_data.append(row)
        
        stats_table = self._create_colorful_table(
            stats_data,
            col_widths=[1.2*inch, 1.8*inch, 1.8*inch, 1.8*inch],
            header_color=colors.HexColor('#7b1fa2'),
            row_colors=[colors.HexColor('#f3e5f5'), colors.HexColor('#e1bee7')]
        )
        story.append(stats_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("<b>Understanding These Statistics:</b>", self.subheading_style))
        stat_explanations = {
            'count': 'Number of non-missing values - tells us how many data points we have',
            'mean': 'Average value (sum of all values divided by count) - the typical value',
            'std': 'Standard deviation - measures how spread out the values are (higher = more variability)',
            'min': 'Minimum value observed - the lowest value in the dataset',
            '25%': 'First quartile - 25% of values are below this number',
            '50%': 'Median - the middle value when data is sorted (50% above, 50% below)',
            '75%': 'Third quartile - 75% of values are below this number',
            'max': 'Maximum value observed - the highest value in the dataset'
        }
        for stat_name, explanation in stat_explanations.items():
            story.append(Paragraph(f"<b>{stat_name.upper()}:</b> {explanation}", self.body_style))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("<b>Key Observations from Statistics:</b>", self.subheading_style))
        median = self.df['wind_generation_actual'].median()
        mean = self.df['wind_generation_actual'].mean()
        observations = [
            f"• Wind generation ranges from {self.df['wind_generation_actual'].min():,.0f} MW to {self.df['wind_generation_actual'].max():,.0f} MW, showing significant variability",
            f"• The median ({median:,.0f} MW) is {'lower' if median < mean else 'higher'} than the mean ({mean:,.0f} MW), indicating {'left' if median > mean else 'right'}-skewed distribution",
            f"• Standard deviation of {self.df['wind_generation_actual'].std():,.0f} MW shows high day-to-day variability in wind generation",
            f"• Temperature ranges from {self.df['temperature'].min():.1f}°C to {self.df['temperature'].max():.1f}°C, typical for German climate",
        ]
        for obs in observations:
            story.append(Paragraph(obs, self.body_style))
        story.append(PageBreak())
        
        # 7. DATA QUALITY ASSESSMENT
        story.append(Paragraph("7. Data Quality Assessment", self.heading_style))
        story.append(Paragraph(
            "Data quality is crucial for building reliable machine learning models. We assess completeness, "
            "consistency, and validity of the data. High-quality data leads to better model performance.",
            self.body_style
        ))
        story.append(Spacer(1, 0.15*inch))
        
        missing = self.df.isnull().sum()
        missing_data = [['Column Name', 'Missing Values', 'Percentage', 'Data Quality']]
        total_rows = len(self.df)
        for col in self.df.columns:
            missing_count = missing[col]
            missing_pct = (missing_count / total_rows) * 100
            quality = "Excellent" if missing_pct == 0 else "Good" if missing_pct < 5 else "Needs Attention"
            missing_data.append([
                col.replace('_', ' ').title(),
                f"{missing_count:,}",
                f"{missing_pct:.2f}%",
                quality
            ])
        
        missing_table = self._create_colorful_table(
            missing_data,
            col_widths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch],
            header_color=colors.HexColor('#c62828'),
            row_colors=[colors.HexColor('#ffebee'), colors.HexColor('#ffcdd2')]
        )
        story.append(missing_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("<b>What Does Data Quality Mean?</b>", self.subheading_style))
        story.append(Paragraph(
            "Missing values represent data points that were not recorded or are unavailable. High percentages "
            "of missing data can reduce model accuracy because the model has less information to learn from. "
            "Our dataset shows excellent data quality with minimal or no missing values, which is ideal for "
            "machine learning. This means we can use almost all of our data for training models without having "
            "to remove many records.",
            self.body_style
        ))
        story.append(PageBreak())
        
        # Key Insights
        story.append(Paragraph("Key Insights", self.heading_style))
        insights = [
            f"• Average daily wind generation: {self.df['wind_generation_actual'].mean():,.0f} MW",
            f"• Maximum wind generation: {self.df['wind_generation_actual'].max():,.0f} MW",
            f"• Minimum wind generation: {self.df['wind_generation_actual'].min():,.0f} MW",
            f"• Average wind capacity: {self.df['wind_capacity'].mean():,.0f} MW",
            f"• Average temperature: {self.df['temperature'].mean():.2f} °C",
            f"• Capacity utilization (avg): {(self.df['wind_generation_actual'] / self.df['wind_capacity']).mean() * 100:.2f}%",
        ]
        for insight in insights:
            story.append(Paragraph(insight, self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Add student footer at the end
        self._add_student_footer(story)
        
        doc.build(story)
        print(f"[OK] Data Overview Report saved to {filename}")
    
    def generate_data_analysis_report(self):
        """Generate Data Analysis Report"""
        print("Generating Data Analysis Report...")
        filename = self.output_dir / "02_Data_Analysis_Report.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=A4)
        story = []
        
        # Add University Logo
        self._add_logo(story)
        
        # Title
        story.append(Paragraph("Wind Energy Forecasting Project", self.title_style))
        story.append(Paragraph("Data Analysis Report", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.heading_style))
        story.append(Paragraph(
            "This report presents a comprehensive analysis of the wind energy dataset, including correlation "
            "analysis, distribution patterns, seasonal trends, and feature engineering insights. The analysis "
            "reveals important patterns and relationships that inform model development and feature selection.",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Correlation Analysis
        story.append(Paragraph("Correlation Analysis", self.heading_style))
        
        corr_matrix = self.df[['wind_generation_actual', 'wind_capacity', 'temperature']].corr()
        corr_data = [['Variable'] + corr_matrix.columns.tolist()]
        for idx in corr_matrix.index:
            row = [idx] + [f"{val:.3f}" for val in corr_matrix.loc[idx].values]
            corr_data.append(row)
        
        corr_table = Table(corr_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        corr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(corr_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Correlation Insights
        story.append(Paragraph("Correlation Insights", self.subheading_style))
        corr_wind_cap = corr_matrix.loc['wind_generation_actual', 'wind_capacity']
        corr_wind_temp = corr_matrix.loc['wind_generation_actual', 'temperature']
        corr_cap_temp = corr_matrix.loc['wind_capacity', 'temperature']
        
        story.append(Paragraph(
            f"• Wind Generation vs Wind Capacity: {corr_wind_cap:.3f} - "
            f"{'Strong' if abs(corr_wind_cap) > 0.7 else 'Moderate' if abs(corr_wind_cap) > 0.4 else 'Weak'} "
            f"{'positive' if corr_wind_cap > 0 else 'negative'} correlation",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"• Wind Generation vs Temperature: {corr_wind_temp:.3f} - "
            f"{'Strong' if abs(corr_wind_temp) > 0.7 else 'Moderate' if abs(corr_wind_temp) > 0.4 else 'Weak'} "
            f"{'positive' if corr_wind_temp > 0 else 'negative'} correlation",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Distribution Analysis
        story.append(Paragraph("Distribution Analysis", self.heading_style))
        
        dist_data = [['Variable', 'Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']]
        for col in ['wind_generation_actual', 'wind_capacity', 'temperature']:
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            std_val = self.df[col].std()
            skew_val = self.df[col].skew()
            kurt_val = self.df[col].kurtosis()
            dist_data.append([
                col.replace('_', ' ').title(),
                f"{mean_val:,.2f}",
                f"{median_val:,.2f}",
                f"{std_val:,.2f}",
                f"{skew_val:.3f}",
                f"{kurt_val:.3f}"
            ])
        
        dist_table = Table(dist_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(dist_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Seasonal Patterns
        story.append(Paragraph("Seasonal Patterns", self.heading_style))
        
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        monthly_avg = self.df.groupby('month')['wind_generation_actual'].mean()
        
        seasonal_data = [['Month', 'Average Wind Generation (MW)', 'Std Dev (MW)']]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in range(1, 13):
            month_data = self.df[self.df['month'] == month]['wind_generation_actual']
            seasonal_data.append([
                month_names[month-1],
                f"{monthly_avg[month]:,.0f}",
                f"{month_data.std():,.0f}"
            ])
        
        seasonal_table = Table(seasonal_data, colWidths=[1.5*inch, 2.5*inch, 2*inch])
        seasonal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(seasonal_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Feature Engineering Summary
        story.append(Paragraph("Feature Engineering Summary", self.heading_style))
        
        feature_cols = FeatureEngineer().get_feature_names(self.df_processed)
        feature_types = {
            'Lag Features': [col for col in feature_cols if 'lag' in col],
            'Rolling Features': [col for col in feature_cols if 'rolling' in col],
            'Seasonal Features': [col for col in feature_cols if any(x in col for x in ['month', 'day', 'week', 'quarter', 'sin', 'cos'])],
            'Temperature Features': [col for col in feature_cols if 'temperature' in col],
            'Capacity Features': [col for col in feature_cols if 'capacity' in col],
        }
        
        for feat_type, features in feature_types.items():
            story.append(Paragraph(f"{feat_type}: {len(features)} features", self.subheading_style))
            if features:
                story.append(Paragraph(f"Examples: {', '.join(features[:5])}", self.styles['Normal']))
                if len(features) > 5:
                    story.append(Paragraph(f"... and {len(features) - 5} more", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph(f"Total Features Created: {len(feature_cols)}", self.subheading_style))
        
        # Add student footer at the end
        self._add_student_footer(story)
        
        doc.build(story)
        print(f"[OK] Data Analysis Report saved to {filename}")
    
    def generate_model_training_report(self):
        """Generate Model Training Report"""
        print("Generating Model Training Report...")
        filename = self.output_dir / "03_Model_Training_Report.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=A4)
        story = []
        
        # Add University Logo
        self._add_logo(story)
        
        # Title
        story.append(Paragraph("Wind Energy Forecasting Project", self.title_style))
        story.append(Paragraph("Model Training Report", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.heading_style))
        story.append(Paragraph(
            "This report documents the training process, architecture, and hyperparameters for all machine "
            "learning models developed for wind energy forecasting. Five different model types were trained: "
            "LSTM, Transformer, XGBoost, LightGBM, and Prophet. Each model was trained on the same dataset "
            "with appropriate preprocessing and feature engineering.",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Training Configuration
        story.append(Paragraph("Training Configuration", self.heading_style))
        
        train_config = [
            ['Parameter', 'Value'],
            ['Train-Test Split', f"{self.config['data']['train_test_split'] * 100:.0f}% / {100 - self.config['data']['train_test_split'] * 100:.0f}%"],
            ['Validation Split', f"{self.config['data']['validation_split'] * 100:.0f}%"],
            ['Target Column', self.config['training']['target_column']],
            ['Random State', str(self.config['training']['random_state'])],
        ]
        
        config_table = Table(train_config, colWidths=[2.5*inch, 3.5*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(config_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Model Details
        models_info = [
            {
                'name': 'LSTM',
                'type': 'Deep Learning - Recurrent Neural Network',
                'config': self.config['models']['lstm'],
                'description': 'Long Short-Term Memory network for sequence learning'
            },
            {
                'name': 'Transformer',
                'type': 'Deep Learning - Attention Mechanism',
                'config': self.config['models']['transformer'],
                'description': 'Transformer architecture with multi-head attention'
            },
            {
                'name': 'XGBoost',
                'type': 'Gradient Boosting',
                'config': self.config['models']['xgboost'],
                'description': 'Extreme Gradient Boosting ensemble method'
            },
            {
                'name': 'LightGBM',
                'type': 'Gradient Boosting',
                'config': self.config['models']['lightgbm'],
                'description': 'Light Gradient Boosting Machine for fast training'
            },
            {
                'name': 'Prophet',
                'type': 'Time Series Forecasting',
                'config': self.config['models']['prophet'],
                'description': "Facebook's Prophet for time series with seasonality"
            },
        ]
        
        for model_info in models_info:
            story.append(Paragraph(f"{model_info['name']} Model", self.heading_style))
            story.append(Paragraph(f"Type: {model_info['type']}", self.subheading_style))
            story.append(Paragraph(f"Description: {model_info['description']}", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            # Hyperparameters
            story.append(Paragraph("Hyperparameters", self.subheading_style))
            params_data = [['Parameter', 'Value']]
            for key, value in model_info['config'].items():
                params_data.append([key.replace('_', ' ').title(), str(value)])
            
            params_table = Table(params_data, colWidths=[2.5*inch, 3.5*inch])
            params_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            story.append(params_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Add student footer at the end
        self._add_student_footer(story)
        
        doc.build(story)
        print(f"[OK] Model Training Report saved to {filename}")
    
    def generate_prediction_report(self):
        """Generate Prediction Report"""
        print("Generating Prediction Report...")
        filename = self.output_dir / "04_Prediction_Report.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=A4)
        story = []
        
        # Add University Logo
        self._add_logo(story)
        
        # Title
        story.append(Paragraph("Wind Energy Forecasting Project", self.title_style))
        story.append(Paragraph("Prediction & Performance Report", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.heading_style))
        story.append(Paragraph(
            "This report presents the prediction results and performance metrics for all trained models. "
            "Each model was evaluated on a held-out test set using multiple metrics including RMSE, MAE, "
            "MAPE, and R² score. The report also includes analysis of prediction accuracy and model comparison.",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        if self.metrics_df is not None:
            # Performance Metrics
            story.append(Paragraph("Model Performance Metrics", self.heading_style))
            
            metrics_data = [['Model'] + self.metrics_df.columns.tolist()]
            for idx in self.metrics_df.index:
                row = [idx]
                for col in self.metrics_df.columns:
                    val = self.metrics_df.loc[idx, col]
                    if col in ['RMSE', 'MAE', 'MSE', 'Std Error']:
                        row.append(f"{val:,.2f}")
                    elif col == 'MAPE':
                        row.append(f"{val:.2f}%")
                    elif col == 'R2':
                        row.append(f"{val:.4f}")
                    else:
                        row.append(f"{val:.2f}")
                metrics_data.append(row)
            
            metrics_table = Table(metrics_data, colWidths=[1.2*inch] + [0.9*inch] * (len(self.metrics_df.columns)))
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Best Model Analysis
            story.append(Paragraph("Best Model Analysis", self.heading_style))
            
            best_rmse = self.metrics_df['RMSE'].idxmin()
            best_mae = self.metrics_df['MAE'].idxmin()
            best_r2 = self.metrics_df['R2'].idxmax()
            
            best_models_data = [
                ['Metric', 'Best Model', 'Value'],
                ['Lowest RMSE', best_rmse, f"{self.metrics_df.loc[best_rmse, 'RMSE']:,.2f}"],
                ['Lowest MAE', best_mae, f"{self.metrics_df.loc[best_mae, 'MAE']:,.2f}"],
                ['Highest R²', best_r2, f"{self.metrics_df.loc[best_r2, 'R2']:.4f}"],
            ]
            
            best_table = Table(best_models_data, colWidths=[2*inch, 2*inch, 2*inch])
            best_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            story.append(best_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Prediction Files Summary
        story.append(Paragraph("Prediction Outputs", self.heading_style))
        
        predictions_dir = Path("outputs/predictions")
        if predictions_dir.exists():
            pred_files = list(predictions_dir.glob("*_predictions.csv"))
            
            pred_summary = [['Model', 'File', 'Records']]
            for pred_file in pred_files:
                try:
                    pred_df = pd.read_csv(pred_file)
                    model_name = pred_file.stem.replace('_predictions', '').title()
                    pred_summary.append([model_name, pred_file.name, f"{len(pred_df):,}"])
                except:
                    pass
            
            if len(pred_summary) > 1:
                pred_table = Table(pred_summary, colWidths=[1.5*inch, 3*inch, 1.5*inch])
                pred_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                story.append(pred_table)
        
        # Add student footer at the end
        self._add_student_footer(story)
        
        doc.build(story)
        print(f"[OK] Prediction Report saved to {filename}")
    
    def generate_dashboard_report(self):
        """Generate Dashboard Report"""
        print("Generating Dashboard Report...")
        filename = self.output_dir / "05_Dashboard_Report.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=A4)
        story = []
        
        # Add University Logo
        self._add_logo(story)
        
        # Title
        story.append(Paragraph("Wind Energy Forecasting Project", self.title_style))
        story.append(Paragraph("Interactive Dashboard Report", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.heading_style))
        story.append(Paragraph(
            "This report documents the interactive Streamlit dashboard developed for the Wind Energy "
            "Forecasting project. The dashboard provides a comprehensive interface for data exploration, "
            "model training status, prediction visualization, and performance comparison. It serves as "
            "the primary user interface for interacting with the forecasting system.",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Dashboard Overview
        story.append(Paragraph("Dashboard Overview", self.heading_style))
        
        overview_data = [
            ['Attribute', 'Value'],
            ['Framework', 'Streamlit'],
            ['Visualization Library', 'Plotly'],
            ['Port', str(self.config['dashboard']['port'])],
            ['Host', self.config['dashboard']['host']],
            ['Theme', self.config['dashboard']['theme']],
            ['Total Pages', '6'],
        ]
        
        overview_table = Table(overview_data, colWidths=[2.5*inch, 3.5*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Dashboard Pages
        story.append(Paragraph("Dashboard Pages", self.heading_style))
        
        pages_info = [
            {
                'name': '📊 Data Overview',
                'description': 'Comprehensive view of the dataset with time series visualizations, statistics, and raw data exploration',
                'features': [
                    'Total records and date range metrics',
                    'Interactive time series plots (All variables, Wind Generation, Wind Capacity, Temperature)',
                    'Statistical summary table',
                    'Raw data table with expandable view'
                ]
            },
            {
                'name': '🔍 Data Analysis',
                'description': 'Deep dive into data patterns, correlations, and distributions',
                'features': [
                    'Correlation matrix heatmap',
                    'Distribution histograms for key variables',
                    'Seasonal pattern analysis by month',
                    'Statistical distribution metrics'
                ]
            },
            {
                'name': '🤖 Model Training',
                'description': 'Interface for model training and status monitoring',
                'features': [
                    'Model selection for training',
                    'Training status indicators',
                    'Model availability checker',
                    'Training instructions and guidance'
                ]
            },
            {
                'name': '📈 Predictions',
                'description': 'Visualize model predictions and performance metrics',
                'features': [
                    'Model selection dropdown',
                    'Interactive predictions vs actual plots',
                    'Residual analysis charts',
                    'Performance metrics display (RMSE, MAE, MAPE, R²)',
                    'Download predictions as CSV'
                ]
            },
            {
                'name': '🔮 Future Forecast',
                'description': 'Generate and visualize future wind energy forecasts',
                'features': [
                    'Model selection for forecasting',
                    'Customizable forecast period (1-365 days)',
                    'Interactive forecast visualization',
                    'Historical context overlay',
                    'Forecast statistics and insights',
                    'Download forecast data'
                ]
            },
            {
                'name': '📉 Model Comparison',
                'description': 'Compare performance across all models',
                'features': [
                    'Comprehensive metrics comparison table',
                    'RMSE comparison bar chart',
                    'R² score comparison bar chart',
                    'Best model identification',
                    'Side-by-side performance visualization'
                ]
            },
        ]
        
        for page_info in pages_info:
            story.append(Paragraph(page_info['name'], self.subheading_style))
            story.append(Paragraph(page_info['description'], self.styles['Normal']))
            story.append(Paragraph("Features:", self.styles['Normal']))
            for feature in page_info['features']:
                story.append(Paragraph(f"• {feature}", self.styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
        
        # Technical Details
        story.append(Paragraph("Technical Implementation", self.heading_style))
        
        tech_details = [
            '• Built with Streamlit framework for rapid web app development',
            '• Uses Plotly for interactive, publication-quality visualizations',
            '• Implements data caching for improved performance',
            '• Modular design with reusable plotting functions',
            '• Responsive layout with wide page configuration',
            '• Dark theme template for better visualization',
            '• Session state management for user interactions',
            '• File-based model loading and prediction generation',
        ]
        
        for detail in tech_details:
            story.append(Paragraph(detail, self.styles['Normal']))
        
        # Add student footer at the end
        self._add_student_footer(story)
        
        doc.build(story)
        print(f"[OK] Dashboard Report saved to {filename}")
    
    def generate_all_reports(self):
        """Generate all reports"""
        print("\n" + "="*60)
        print("Generating Comprehensive PDF Reports")
        print("="*60 + "\n")
        
        self.generate_data_overview_report()
        self.generate_data_analysis_report()
        self.generate_model_training_report()
        self.generate_prediction_report()
        self.generate_dashboard_report()
        
        print("\n" + "="*60)
        print("All reports generated successfully!")
        print(f"Reports saved in: {self.output_dir.absolute()}")
        print("="*60)


if __name__ == "__main__":
    generator = PDFReportGenerator()
    generator.generate_all_reports()
