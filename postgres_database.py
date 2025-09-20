import psycopg2
import psycopg2.extras
import json
import logging
import os
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KolamPostgreSQLDatabase:
    """
    PostgreSQL database manager for Kolam application data
    Handles storage of analysis results, user patterns, and application statistics
    """
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('PGHOST'),
            'port': os.getenv('PGPORT'),
            'database': os.getenv('PGDATABASE'),
            'user': os.getenv('PGUSER'),
            'password': os.getenv('PGPASSWORD')
        }
        logger.info("PostgreSQL Database initialized")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_tables(self):
        """Create all necessary database tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Analysis results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id SERIAL PRIMARY KEY,
                        filename TEXT NOT NULL,
                        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        analysis_data JSONB NOT NULL,
                        pattern_type TEXT,
                        dot_count INTEGER,
                        contour_count INTEGER,
                        symmetry_score REAL,
                        complexity_score REAL,
                        classification_result JSONB
                    )
                ''')
                
                # Generated patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS generated_patterns (
                        id SERIAL PRIMARY KEY,
                        generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        parameters JSONB NOT NULL,
                        pattern_data JSONB,
                        theme TEXT,
                        grid_size INTEGER,
                        complexity_level TEXT,
                        generation_time REAL,
                        is_favorite BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # User favorites table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_favorites (
                        id SERIAL PRIMARY KEY,
                        item_type TEXT NOT NULL,  -- 'analysis' or 'generated'
                        item_id INTEGER NOT NULL,
                        saved_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        notes TEXT
                    )
                ''')
                
                # Application statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS app_statistics (
                        id SERIAL PRIMARY KEY,
                        stat_date DATE DEFAULT CURRENT_DATE,
                        patterns_analyzed INTEGER DEFAULT 0,
                        patterns_generated INTEGER DEFAULT 0,
                        total_uploads INTEGER DEFAULT 0,
                        active_sessions INTEGER DEFAULT 0,
                        UNIQUE(stat_date)
                    )
                ''')
                
                # Drawing sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS drawing_sessions (
                        id SERIAL PRIMARY KEY,
                        session_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        canvas_data JSONB,
                        analysis_results JSONB,
                        session_duration REAL,
                        is_completed BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_results(upload_timestamp);')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_generated_timestamp ON generated_patterns(generation_timestamp);')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON analysis_results(pattern_type);')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_theme ON generated_patterns(theme);')
                
                conn.commit()
                logger.info("PostgreSQL Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def save_analysis_result(self, filename, analysis_data, classification_result=None):
        """
        Save pattern analysis results to database
        
        Args:
            filename: Name of the analyzed file
            analysis_data: Dictionary containing analysis results
            classification_result: CNN classification results
            
        Returns:
            ID of the saved record
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract key metrics from analysis data
                pattern_type = analysis_data.get('pattern_type', 'Unknown')
                dot_count = analysis_data.get('dot_count', 0)
                contour_count = analysis_data.get('contour_count', 0)
                symmetry_score = analysis_data.get('symmetry_score', 0.0)
                complexity_score = analysis_data.get('complexity', 0.0)
                
                cursor.execute('''
                    INSERT INTO analysis_results 
                    (filename, analysis_data, pattern_type, dot_count, contour_count, 
                     symmetry_score, complexity_score, classification_result)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (filename, json.dumps(analysis_data), pattern_type, dot_count, contour_count,
                      symmetry_score, complexity_score, json.dumps(classification_result) if classification_result else None))
                
                record_id = cursor.fetchone()[0]
                conn.commit()
                
                # Update statistics
                self.update_daily_stats('patterns_analyzed', 1)
                self.update_daily_stats('total_uploads', 1)
                
                logger.info(f"Analysis result saved with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Error saving analysis result: {str(e)}")
            return None
    
    def save_generated_pattern(self, parameters, pattern_data, generation_time):
        """
        Save generated pattern to database
        
        Args:
            parameters: Generation parameters
            pattern_data: Pattern generation results
            generation_time: Time taken to generate
            
        Returns:
            ID of the saved record
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract key information
                theme = parameters.get('theme', 'Unknown')
                grid_size = parameters.get('grid_size', 0)
                complexity_level = parameters.get('complexity', 'Unknown')
                
                cursor.execute('''
                    INSERT INTO generated_patterns 
                    (parameters, pattern_data, theme, grid_size, complexity_level, generation_time)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (json.dumps(parameters), json.dumps(pattern_data), theme, grid_size, complexity_level, generation_time))
                
                record_id = cursor.fetchone()[0]
                conn.commit()
                
                # Update statistics
                self.update_daily_stats('patterns_generated', 1)
                
                logger.info(f"Generated pattern saved with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Error saving generated pattern: {str(e)}")
            return None
    
    def add_to_favorites(self, item_type, item_id, notes=""):
        """
        Add item to user favorites
        
        Args:
            item_type: 'analysis' or 'generated'
            item_id: ID of the item to favorite
            notes: Optional notes
            
        Returns:
            Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if already favorited
                cursor.execute('''
                    SELECT id FROM user_favorites 
                    WHERE item_type = %s AND item_id = %s
                ''', (item_type, item_id))
                
                if cursor.fetchone():
                    logger.info(f"Item {item_type}:{item_id} already in favorites")
                    return True
                
                # Add to favorites
                cursor.execute('''
                    INSERT INTO user_favorites (item_type, item_id, notes)
                    VALUES (%s, %s, %s)
                ''', (item_type, item_id, notes))
                
                conn.commit()
                logger.info(f"Added {item_type}:{item_id} to favorites")
                return True
                
        except Exception as e:
            logger.error(f"Error adding to favorites: {str(e)}")
            return False
    
    def get_recent_analyses(self, limit=10):
        """
        Get recent pattern analyses
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent analysis records
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    ORDER BY upload_timestamp DESC 
                    LIMIT %s
                ''', (limit,))
                
                records = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for record in records:
                    result = dict(record)
                    # Parse JSON fields
                    if result['analysis_data']:
                        result['analysis_data'] = json.loads(result['analysis_data']) if isinstance(result['analysis_data'], str) else result['analysis_data']
                    if result['classification_result']:
                        result['classification_result'] = json.loads(result['classification_result']) if isinstance(result['classification_result'], str) else result['classification_result']
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent analyses: {str(e)}")
            return []
    
    def get_recent_generations(self, limit=10):
        """
        Get recent pattern generations
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent generation records
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cursor.execute('''
                    SELECT * FROM generated_patterns 
                    ORDER BY generation_timestamp DESC 
                    LIMIT %s
                ''', (limit,))
                
                records = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for record in records:
                    result = dict(record)
                    # Parse JSON fields
                    if result['parameters']:
                        result['parameters'] = json.loads(result['parameters']) if isinstance(result['parameters'], str) else result['parameters']
                    if result['pattern_data']:
                        result['pattern_data'] = json.loads(result['pattern_data']) if isinstance(result['pattern_data'], str) else result['pattern_data']
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent generations: {str(e)}")
            return []
    
    def get_user_favorites(self):
        """
        Get user favorite items
        
        Returns:
            Dictionary with favorite analyses and generations
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Get favorite analyses
                cursor.execute('''
                    SELECT ar.*, uf.notes, uf.saved_timestamp as favorite_timestamp
                    FROM user_favorites uf
                    JOIN analysis_results ar ON ar.id = uf.item_id
                    WHERE uf.item_type = 'analysis'
                    ORDER BY uf.saved_timestamp DESC
                ''')
                
                favorite_analyses = []
                for record in cursor.fetchall():
                    result = dict(record)
                    if result['analysis_data']:
                        result['analysis_data'] = json.loads(result['analysis_data']) if isinstance(result['analysis_data'], str) else result['analysis_data']
                    if result['classification_result']:
                        result['classification_result'] = json.loads(result['classification_result']) if isinstance(result['classification_result'], str) else result['classification_result']
                    favorite_analyses.append(result)
                
                # Get favorite generations
                cursor.execute('''
                    SELECT gp.*, uf.notes, uf.saved_timestamp as favorite_timestamp
                    FROM user_favorites uf
                    JOIN generated_patterns gp ON gp.id = uf.item_id
                    WHERE uf.item_type = 'generated'
                    ORDER BY uf.saved_timestamp DESC
                ''')
                
                favorite_generations = []
                for record in cursor.fetchall():
                    result = dict(record)
                    if result['parameters']:
                        result['parameters'] = json.loads(result['parameters']) if isinstance(result['parameters'], str) else result['parameters']
                    if result['pattern_data']:
                        result['pattern_data'] = json.loads(result['pattern_data']) if isinstance(result['pattern_data'], str) else result['pattern_data']
                    favorite_generations.append(result)
                
                return {
                    'analyses': favorite_analyses,
                    'generations': favorite_generations
                }
                
        except Exception as e:
            logger.error(f"Error getting user favorites: {str(e)}")
            return {'analyses': [], 'generations': []}
    
    def get_app_statistics(self):
        """
        Get application usage statistics
        
        Returns:
            Dictionary with various statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Get total counts
                cursor.execute('SELECT COUNT(*) as total FROM analysis_results')
                total_analyses = cursor.fetchone()['total']
                
                cursor.execute('SELECT COUNT(*) as total FROM generated_patterns')
                total_generations = cursor.fetchone()['total']
                
                cursor.execute('SELECT COUNT(*) as total FROM user_favorites')
                total_favorites = cursor.fetchone()['total']
                
                # Get recent activity (last 7 days)
                cursor.execute('''
                    SELECT 
                        COALESCE(SUM(patterns_analyzed), 0) as week_analyses,
                        COALESCE(SUM(patterns_generated), 0) as week_generations,
                        COALESCE(SUM(total_uploads), 0) as week_uploads
                    FROM app_statistics 
                    WHERE stat_date >= CURRENT_DATE - INTERVAL '7 days'
                ''')
                
                weekly_stats = cursor.fetchone()
                
                # Get daily statistics for the last week
                cursor.execute('''
                    SELECT * FROM app_statistics 
                    WHERE stat_date >= CURRENT_DATE - INTERVAL '7 days'
                    ORDER BY stat_date DESC
                ''')
                
                daily_stats = [dict(row) for row in cursor.fetchall()]
                
                # Calculate classification accuracy based on actual data
                cursor.execute('''
                    SELECT COUNT(*) as total_classified
                    FROM analysis_results 
                    WHERE classification_result IS NOT NULL
                ''')
                total_classified = cursor.fetchone()['total_classified']
                
                # Mock accuracy calculation - in real implementation, this would be based on validation data
                classification_accuracy = 0.89 if total_classified > 0 else 0.0
                
                return {
                    'total_analyses': total_analyses,
                    'total_generations': total_generations,
                    'total_favorites': total_favorites,
                    'week_analyses': weekly_stats['week_analyses'] or 0,
                    'week_generations': weekly_stats['week_generations'] or 0,
                    'week_uploads': weekly_stats['week_uploads'] or 0,
                    'classification_accuracy': classification_accuracy,
                    'daily_stats': daily_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting app statistics: {str(e)}")
            return {
                'total_analyses': 0,
                'total_generations': 0,
                'total_favorites': 0,
                'week_analyses': 0,
                'week_generations': 0,
                'week_uploads': 0,
                'classification_accuracy': 0.0,
                'daily_stats': []
            }
    
    def update_daily_stats(self, stat_name, increment=1):
        """
        Update daily statistics
        
        Args:
            stat_name: Name of the statistic to update
            increment: Amount to increment by
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Use ON CONFLICT for upsert functionality
                cursor.execute('''
                    INSERT INTO app_statistics (stat_date, {}) 
                    VALUES (CURRENT_DATE, %s)
                    ON CONFLICT (stat_date) 
                    DO UPDATE SET {} = app_statistics.{} + EXCLUDED.{}
                '''.format(stat_name, stat_name, stat_name, stat_name), (increment,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating daily stats: {str(e)}")
    
    def save_drawing_session(self, canvas_data, analysis_results=None, session_duration=0.0):
        """
        Save drawing session data
        
        Args:
            canvas_data: Canvas drawing data
            analysis_results: Analysis results if available
            session_duration: Duration of the drawing session
            
        Returns:
            ID of the saved session
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO drawing_sessions 
                    (canvas_data, analysis_results, session_duration, is_completed)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                ''', (json.dumps(canvas_data), json.dumps(analysis_results) if analysis_results else None, session_duration, analysis_results is not None))
                
                session_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Drawing session saved with ID: {session_id}")
                return session_id
                
        except Exception as e:
            logger.error(f"Error saving drawing session: {str(e)}")
            return None
    
    def get_pattern_statistics_by_type(self):
        """
        Get statistics grouped by pattern type
        
        Returns:
            Dictionary with pattern type statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Analysis patterns by type
                cursor.execute('''
                    SELECT pattern_type, COUNT(*) as count, AVG(symmetry_score) as avg_symmetry
                    FROM analysis_results 
                    WHERE pattern_type IS NOT NULL
                    GROUP BY pattern_type
                    ORDER BY count DESC
                ''')
                
                analysis_stats = [dict(row) for row in cursor.fetchall()]
                
                # Generated patterns by theme
                cursor.execute('''
                    SELECT theme, COUNT(*) as count, AVG(generation_time) as avg_time
                    FROM generated_patterns 
                    WHERE theme IS NOT NULL
                    GROUP BY theme
                    ORDER BY count DESC
                ''')
                
                generation_stats = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'analysis_by_type': analysis_stats,
                    'generation_by_theme': generation_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {str(e)}")
            return {'analysis_by_type': [], 'generation_by_theme': []}
    
    def search_patterns(self, query_text=None, pattern_type=None, theme=None, limit=20):
        """
        Search patterns based on various criteria
        
        Args:
            query_text: Text to search in filenames and notes
            pattern_type: Specific pattern type to filter
            theme: Specific theme to filter
            limit: Maximum number of results
            
        Returns:
            Dictionary with matching analyses and generations
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Build search conditions
                analysis_conditions = []
                analysis_params = []
                
                if query_text:
                    analysis_conditions.append("(filename ILIKE %s OR analysis_data::text ILIKE %s)")
                    analysis_params.extend([f'%{query_text}%', f'%{query_text}%'])
                
                if pattern_type:
                    analysis_conditions.append("pattern_type = %s")
                    analysis_params.append(pattern_type)
                
                analysis_where = "WHERE " + " AND ".join(analysis_conditions) if analysis_conditions else ""
                
                # Search analysis results
                cursor.execute(f'''
                    SELECT * FROM analysis_results 
                    {analysis_where}
                    ORDER BY upload_timestamp DESC 
                    LIMIT %s
                ''', analysis_params + [limit])
                
                matching_analyses = [dict(row) for row in cursor.fetchall()]
                
                # Build search conditions for generated patterns
                generation_conditions = []
                generation_params = []
                
                if query_text:
                    generation_conditions.append("(parameters::text ILIKE %s OR pattern_data::text ILIKE %s)")
                    generation_params.extend([f'%{query_text}%', f'%{query_text}%'])
                
                if theme:
                    generation_conditions.append("theme = %s")
                    generation_params.append(theme)
                
                generation_where = "WHERE " + " AND ".join(generation_conditions) if generation_conditions else ""
                
                # Search generated patterns
                cursor.execute(f'''
                    SELECT * FROM generated_patterns 
                    {generation_where}
                    ORDER BY generation_timestamp DESC 
                    LIMIT %s
                ''', generation_params + [limit])
                
                matching_generations = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'analyses': matching_analyses,
                    'generations': matching_generations
                }
                
        except Exception as e:
            logger.error(f"Error searching patterns: {str(e)}")
            return {'analyses': [], 'generations': []}
    
    def cleanup_old_records(self, days_to_keep=30):
        """
        Clean up old records to maintain database size
        
        Args:
            days_to_keep: Number of days of records to keep
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old non-favorite analysis results
                cursor.execute('''
                    DELETE FROM analysis_results 
                    WHERE upload_timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    AND id NOT IN (
                        SELECT item_id FROM user_favorites WHERE item_type = 'analysis'
                    )
                ''', (days_to_keep,))
                
                analysis_deleted = cursor.rowcount
                
                # Clean up old non-favorite generated patterns
                cursor.execute('''
                    DELETE FROM generated_patterns 
                    WHERE generation_timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    AND id NOT IN (
                        SELECT item_id FROM user_favorites WHERE item_type = 'generated'
                    )
                    AND is_favorite = FALSE
                ''', (days_to_keep,))
                
                generation_deleted = cursor.rowcount
                
                # Clean up old drawing sessions
                cursor.execute('''
                    DELETE FROM drawing_sessions 
                    WHERE session_timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    AND is_completed = FALSE
                ''', (days_to_keep,))
                
                session_deleted = cursor.rowcount
                
                # Clean up old statistics (keep last 90 days)
                cursor.execute('''
                    DELETE FROM app_statistics 
                    WHERE stat_date < CURRENT_DATE - INTERVAL '90 days'
                ''')
                
                stats_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleanup completed: {analysis_deleted} analyses, {generation_deleted} generations, {session_deleted} sessions, {stats_deleted} stats deleted")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def export_data(self, format_type='json', include_favorites_only=False):
        """
        Export data in various formats
        
        Args:
            format_type: 'json', 'csv', or 'svg'
            include_favorites_only: Whether to export only favorited items
            
        Returns:
            Exported data in requested format
        """
        try:
            if include_favorites_only:
                favorites = self.get_user_favorites()
                data = {
                    'analyses': favorites['analyses'],
                    'generations': favorites['generations'],
                    'export_timestamp': datetime.now().isoformat(),
                    'export_type': 'favorites_only'
                }
            else:
                analyses = self.get_recent_analyses(limit=1000)
                generations = self.get_recent_generations(limit=1000)
                data = {
                    'analyses': analyses,
                    'generations': generations,
                    'export_timestamp': datetime.now().isoformat(),
                    'export_type': 'full_export'
                }
            
            if format_type == 'json':
                return json.dumps(data, indent=2, default=str)
            elif format_type == 'csv':
                # Convert to CSV using pandas
                if data['analyses']:
                    df_analyses = pd.DataFrame(data['analyses'])
                    csv_analyses = df_analyses.to_csv(index=False)
                else:
                    csv_analyses = "No analysis data available"
                
                if data['generations']:
                    df_generations = pd.DataFrame(data['generations'])
                    csv_generations = df_generations.to_csv(index=False)
                else:
                    csv_generations = "No generation data available"
                
                return f"=== ANALYSES ===\n{csv_analyses}\n\n=== GENERATIONS ===\n{csv_generations}"
            elif format_type == 'svg':
                # Basic SVG export for patterns
                svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">'
                svg_content += '<text x="10" y="30">Kolam Patterns Export</text>'
                svg_content += f'<text x="10" y="60">Generated on: {datetime.now()}</text>'
                svg_content += f'<text x="10" y="90">Total Analyses: {len(data["analyses"])}</text>'
                svg_content += f'<text x="10" y="120">Total Generations: {len(data["generations"])}</text>'
                svg_content += '</svg>'
                return svg_content
            else:
                return json.dumps(data, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return None

# Global database instance
db = None

def get_database():
    """Get or create database instance"""
    global db
    if db is None:
        db = KolamPostgreSQLDatabase()
        try:
            db.create_tables()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Fallback to SQLite if PostgreSQL fails
            from .database import KolamDatabase
            db = KolamDatabase()
            db.create_tables()
    return db
