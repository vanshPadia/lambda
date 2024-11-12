import os
import io
import smtplib
import logging
import mysql.connector
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
import seaborn as sns
import tempfile

# Configure matplotlib for non-interactive backend
matplotlib.use('Agg')

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def validate_email_config(sender_email, sender_password, recipient_group):
    """Validate email configuration before attempting to send"""
    if not all([sender_email, sender_password]):
        raise ValueError("Missing email credentials")
    if not recipient_group:
        raise ValueError("No recipient emails configured")

def test_smtp_connection(smtp_server, smtp_port, sender_email, sender_password):
    """Test SMTP connection to verify email server availability"""
    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(sender_email, sender_password)
        logger.info("SMTP connection test successful")
        return True
    except Exception as e:
        logger.error(f"SMTP connection test failed: {str(e)}")
        return False

def fetch_data_from_db(db_host, db_user, db_password, db_name):
    """Fetch data from MySQL database for the report"""
    logger.info("Fetching data from database...")
    connection = None
    try:
        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            connect_timeout=10
        )
        
        query = """
        SELECT 
            pub.name AS publisher_name,
            WEEK(asm.creation_date, 1) AS week_no,
            DATE(DATE_SUB(asm.creation_date, INTERVAL WEEKDAY(asm.creation_date) DAY)) AS week_start_date, 
            COUNT(DISTINCT asmd.id) AS modal_display_count,
            COUNT(DISTINCT asma.id) AS modal_accept_count,
            COUNT(DISTINCT aasus.user_id) AS user_count,
            CASE 
                WHEN COUNT(DISTINCT asmd.id) > 0 THEN 
                    (COUNT(DISTINCT asma.id) / COUNT(DISTINCT asmd.id)) * 100 
                ELSE 0 
            END AS modal_accept_percentage,
            CASE 
                WHEN COUNT(DISTINCT aasus.user_id) > 0 THEN 
                    COUNT(DISTINCT asma.id) / COUNT(DISTINCT aasus.user_id) 
                ELSE 0 
            END AS avg_modal_accepted_per_user,
            IFNULL(estimated_views.estimated_views / 2, 0) AS estimated_views
        FROM 
            adn_am_ad_serving_modal asm
        JOIN 
            adn_am_publisher pub ON asm.publisher_id = pub.id
        LEFT JOIN 
            adn_am_ad_serving_modal_display asmd ON asm.id = asmd.ad_serving_modal_id
        LEFT JOIN 
            adn_am_ad_serving_modal_accept asma ON asm.id = asma.ad_serving_modal_id
        JOIN 
            adn_am_sso_user_session aasus ON asm.user_session_id = aasus.id
        LEFT JOIN 
            adn_um_sso_user usr ON usr.id = aasus.user_id
        LEFT JOIN (
            SELECT 
                WEEK(h.creation_date, 1) AS week_no,
                DATE(DATE_SUB(h.creation_date, INTERVAL WEEKDAY(h.creation_date) DAY)) AS week_start_date,
                COUNT(*) AS estimated_views
            FROM 
                adn_am_publisher_content_log_history h
            WHERE 
                h.ip_address NOT IN (
                    SELECT specific_ip_address 
                    FROM adn_am_sso_test_user_identity_config 
                    WHERE specific_ip_address IS NOT NULL
                )
                AND h.view_type = 'MIFAN_BTN_INIT'
                AND h.publisher_id = 18
                AND DATE(h.creation_date) >= '2024-06-17' 
                AND DATE(h.creation_date) <= '2024-09-30'
            GROUP BY 
                WEEK(h.creation_date, 1),
                DATE(DATE_SUB(h.creation_date, INTERVAL WEEKDAY(h.creation_date) DAY))
        ) AS estimated_views 
        ON WEEK(asm.creation_date, 1) = estimated_views.week_no 
        AND DATE(DATE_SUB(asm.creation_date, INTERVAL WEEKDAY(asm.creation_date) DAY)) = estimated_views.week_start_date
        WHERE 
            asm.creation_date >= '2024-06-17' 
            AND asm.creation_date < '2024-09-30'
            AND aasus.is_test_user = 0
            AND pub.id = 17
            AND COALESCE(aasus.user_id, 0) NOT IN (
                SELECT id 
                FROM adn_um_sso_user usr 
                WHERE email LIKE '%shantel%'
            )
        GROUP BY 
            pub.name,
            week_no,
            week_start_date
        ORDER BY 
            week_start_date ASC;
        """
        
        df = pd.read_sql(query, connection)
        logger.info(f"Successfully fetched {len(df)} rows of data")
        return df
        
    except mysql.connector.Error as err:
        logger.error(f"Database error: {str(err)}")
        raise
    finally:
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")

def create_graph(df):
    """Create visualization graphs with memory optimization"""
    logger.info("Creating updated visualization graphs...")
    try:
        # Convert week_start_date to datetime
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])

        sns.set(style='whitegrid')
        plt.figure(figsize=(14, 12))

        # Line plots
        line_metrics = ['modal_display_count', 'modal_accept_count', 'user_count',
                        'modal_accept_percentage', 'avg_modal_accepted_per_user', 'estimated_views']
        for i, line_metric in enumerate(line_metrics, 1):
            plt.subplot(3, 2, i)
            sns.lineplot(data=df, x='week_no', y=line_metric, marker='o')
            plt.title(f'{line_metric.replace("_", " ").title()} over Weeks')
            plt.xlabel('Week Number')
            plt.ylabel(line_metric.replace("_", " ").title())

        plt.tight_layout()
        line_chart_buffer = io.BytesIO()
        plt.savefig(line_chart_buffer, format='png', dpi=300, bbox_inches='tight')
        line_chart_buffer.seek(0)
        plt.close()

        # Bar plots
        plt.figure(figsize=(12, 10))
        bar_metrics = [('modal_display_count', 'Blues'), ('modal_accept_count', 'Oranges'),
                       ('user_count', 'Greens'), ('estimated_views', 'Reds')]
        for i, (bar_metric, color) in enumerate(bar_metrics, 1):
            plt.subplot(2, 2, i)
            sns.barplot(data=df, x='week_no', y=bar_metric, palette=color)
            plt.title(f'{bar_metric.replace("_", " ").title()} by Week')
            plt.xlabel('Week Number')
            plt.ylabel(bar_metric.replace("_", " ").title())

        plt.tight_layout()
        bar_chart_buffer = io.BytesIO()
        plt.savefig(bar_chart_buffer, format='png', dpi=300, bbox_inches='tight')
        bar_chart_buffer.seek(0)
        plt.close()

        # Comparison Bar Plot for modal displays and accepts across weeks
        plt.figure(figsize=(15, 8))
        width = 0.3
        x = df['week_no']

        # Bar plots for modal display and accept counts
        plt.bar(x - width / 2, df['modal_display_count'], width, label='Modal Display Count', color='blue')
        plt.bar(x + width / 2, df['modal_accept_count'], width, label='Modal Accept Count', color='orange')

        plt.xlabel('Week Number')
        plt.ylabel('Count')
        plt.title('Comparison of Modal Displays and Accepts per Week')
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        comparison_bar_buffer = io.BytesIO()
        plt.savefig(comparison_bar_buffer, format='png', dpi=300, bbox_inches='tight')
        comparison_bar_buffer.seek(0)
        plt.close()

        # Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[['modal_display_count', 'modal_accept_count', 'user_count',
                                 'modal_accept_percentage', 'avg_modal_accepted_per_user', 'estimated_views']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Key Metrics')
        plt.tight_layout()
        heatmap_buffer = io.BytesIO()
        plt.savefig(heatmap_buffer, format='png', dpi=300, bbox_inches='tight')
        heatmap_buffer.seek(0)
        plt.close()

        # Pie chart for a specific week
        week_data = df[df['week_no'] == 25].iloc[0]
        plt.figure(figsize=(8, 6))
        plt.pie(week_data[['modal_display_count', 'modal_accept_count', 'user_count', 'estimated_views']],
                labels=['Modal Display Count', 'Modal Accept Count', 'User Count', 'Estimated Views'],
                autopct='%1.1f%%', startangle=140, colors=['blue', 'orange', 'green', 'red'])
        plt.title('Breakdown of Metrics for Week 25')
        pie_chart_buffer = io.BytesIO()
        plt.savefig(pie_chart_buffer, format='png', dpi=300, bbox_inches='tight')
        pie_chart_buffer.seek(0)
        plt.close()

        # Return the buffers for each chart
        return {
            "line_chart": line_chart_buffer,
            "bar_chart": bar_chart_buffer,
            "comparison_bar": comparison_bar_buffer,
            "heatmap": heatmap_buffer,
            "pie_chart": pie_chart_buffer
        }
        
    except Exception as e:
        logger.error(f"Error creating graphs: {str(e)}")
        raise
    finally:
        plt.close('all')

def send_report(df, smtp_server, smtp_port, sender_email, sender_password, recipient_group):
    """Send email report with graphs and data"""
    logger.info("Preparing email report...")
    try:
        validate_email_config(sender_email, sender_password, recipient_group)
        
        msg = MIMEMultipart()
        msg['Subject'] = f"MIFAN Weekly Report - {datetime.now().strftime('%Y-%m-%d')}"
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipient_group)

        # Create HTML body
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>MIFAN Weekly Report</h2>
            <p>Please find attached the weekly metrics report and visualizations.</p>
            
            <h3>Summary Statistics:</h3>
            <ul>
                <li>Total Modal Displays: <strong>{df['modal_display_count'].sum():,}</strong></li>
                <li>Total Modal Accepts: <strong>{df['modal_accept_count'].sum():,}</strong></li>
                <li>Average Weekly Users: <strong>{df['user_count'].mean():,.0f}</strong></li>
                <li>Overall Acceptance Rate: <strong>{(df['modal_accept_count'].sum() / df['modal_display_count'].sum() * 100):.1f}%</strong></li>
            </ul>
            
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        # Attach CSV report
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_attachment = MIMEApplication(csv_buffer.getvalue().encode('utf-8'), _subtype='csv')
        csv_attachment.add_header('Content-Disposition', 'attachment', 
                                filename=f'mifan_report_{datetime.now().strftime("%Y%m%d")}.csv')
        msg.attach(csv_attachment)

        # Attach graphs
        graph_buffers = create_graph(df)
        for graph_name, graph_buffer in graph_buffers.items():
            graph_attachment = MIMEApplication(graph_buffer.getvalue(), _subtype='png')
            graph_attachment.add_header('Content-Disposition', 'attachment', 
                                     filename=f'{graph_name}_{datetime.now().strftime("%Y%m%d")}.png')
            msg.attach(graph_attachment)

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_group, msg.as_string())
            
        logger.info("Email report sent successfully")

    except Exception as e:
        logger.error(f"Error sending report: {str(e)}")
        raise

def lambda_handler(event, context):
    """Main Lambda handler function"""
    try:
        logger.info("Starting MIFAN report generation...")
        
        # Get environment variables
        env_vars = {
            'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.environ.get('SMTP_PORT', '587')),
            'sender_email': os.environ.get('SENDER_EMAIL'),
            'sender_password': os.environ.get('SENDER_PASSWORD'),
            'recipient_group': os.environ.get('RECIPIENT_EMAILS', '').split(','),
            'db_host': os.environ.get('DB_HOST'),
            'db_user': os.environ.get('DB_USER'),
            'db_password': os.environ.get('DB_PASSWORD'),
            'db_name': os.environ.get('DB_NAME')
        }

        # Validate environment variables
        missing_vars = [k for k, v in env_vars.items() if not v]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Set up matplotlib temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ['MPLCONFIGDIR'] = tmp_dir
            
            # Fetch data
            df = fetch_data_from_db(
                env_vars['db_host'],
                env_vars['db_user'],
                env_vars['db_password'],
                env_vars['db_name']
            )
            
            if df.empty:
                raise ValueError("No data retrieved from database")

            # Send report
            send_report(
                df,
                env_vars['smtp_server'],
                env_vars['smtp_port'],
                env_vars['sender_email'],
                env_vars['sender_password'],
                env_vars['recipient_group']
            )

        return {
            'statusCode': 200,
            'body': 'Report generated and sent successfully'
        }

    except Exception as e:
        error_msg = f"Lambda execution failed: {str(e)}"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': error_msg
        }
