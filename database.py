import sqlite3
import json
import logging
import os
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KolamDatabase:
    """
    Database manager for Kolam application data
    Handles storage of analysis results, user patterns, and application statistics
    """
    
    def __init__(self, db_path="data/kolam_app.db"):
        self.db_path = db_path
        self.ensure_db_directory()
        logger.info(f"Database initialized at {db_path}")
    
    def ensure_db_directory(self):
        """Ensure the database directory exists"""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
        except Exception as e:
            logger.error(f"Error creating database directory: {str(e)}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
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
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        analysis_data TEXT NOT NULL,
                        pattern_type TEXT,
                        dot_count INTEGER,
                        contour_count INTEGER,
                        symmetry_score REAL,
                        complexity_score REAL,
                        classification_result TEXT
                    )
                ''')
                
                # Generated patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS generated_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        generation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        parameters TEXT NOT NULL,
                        pattern_data TEXT,
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
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_type TEXT NOT NULL,  -- 'analysis' or 'generated'
                        item_id INTEGER NOT NULL,
                        saved_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        notes TEXT
                    )
                ''')
                
                # Application statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS app_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stat_date DATE DEFAULT (date('now')),
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
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        canvas_data TEXT,
                        analysis_results TEXT,
                        session_duration REAL,
                        is_completed BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                conn.commit()
                logger.info("Database tables created successfully")
                
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
                
                # Convert dictionaries to JSON strings
                analysis_json = json.dumps(analysis_data, default=str)
                classification_json = json.dumps(classification_result, default=str) if classification_result else None
                
                cursor.execute('''
                    INSERT INTO analysis_results 
                    (filename, analysis_data, pattern_type, dot_count, contour_count, 
                     symmetry_score, complexity_score, classification_result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (filename, analysis_json, pattern_type, dot_count, contour_count,
                      symmetry_score, complexity_score, classification_json))
                
                record_id = cursor.lastrowid
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
                
                # Convert to JSON
                parameters_json = json.dumps(parameters, default=str)
                pattern_json = json.dumps(pattern_data, default=str)
                
                cursor.execute('''
                    INSERT INTO generated_patterns 
                    (parameters, pattern_data, theme, grid_size, complexity_level, generation_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (parameters_json, pattern_json, theme, grid_size, complexity_level, generation_time))
                
                record_id = cursor.lastrowid
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
                    WHERE item_type = ? AND item_id = ?
                ''', (item_type, item_id))
                
                if cursor.fetchone():
                    logger.info(f"Item {item_type}:{item_id} already in favorites")
                    return True
                
                # Add to favorites
                cursor.execute('''
                    INSERT INTO user_favorites (item_type, item_id, notes)
                    VALUES (?, ?, ?)
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
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    ORDER BY upload_timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                records = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for record in records:
                    result = dict(record)
                    # Parse JSON fields
                    if result['analysis_data']:
                        result['analysis_data'] = json.loads(result['analysis_data'])
                    if result['classification_result']:
                        result['classification_result'] = json.loads(result['classification_result'])
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
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM generated_patterns 
                    ORDER BY generation_timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                records = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for record in records:
                    result = dict(record)
                    # Parse JSON fields
                    if result['parameters']:
                        result['parameters'] = json.loads(result['parameters'])
                    if result['pattern_data']:
                        result['pattern_data'] = json.loads(result['pattern_data'])
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
                cursor = conn.cursor()
                
                # Get favorite analyses
                cursor.execute('''
                    SELECT ar.*, uf.notes, uf.saved_timestamp
                    FROM user_favorites uf
                    JOIN analysis_results ar ON ar.id = uf.item_id
                    WHERE uf.item_type = 'analysis'
                    ORDER BY uf.saved_timestamp DESC
                ''')
                
                favorite_analyses = []
                for record in cursor.fetchall():
                    result = dict(record)
                    if result['analysis_data']:
                        result['analysis_data'] = json.loads(result['analysis_data'])
                    if result['classification_result']:
                        result['classification_result'] = json.loads(result['classification_result'])
                    favorite_analyses.append(result)
                
                # Get favorite generations
                cursor.execute('''
                    SELECT gp.*, uf.notes, uf.saved_timestamp
                    FROM user_favorites uf
                    JOIN generated_patterns gp ON gp.id = uf.item_id
                    WHERE uf.item_type = 'generated'
                    ORDER BY uf.saved_timestamp DESC
                ''')
                
                favorite_generations = []
                for record in cursor.fetchall():
                    result = dict(record)
                    if result['parameters']:
                        result['parameters'] = json.loads(result['parameters'])
                    if result['pattern_data']:
                        result['pattern_data'] = json.loads(result['pattern_data'])
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
                cursor = conn.cursor()
                
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
                        SUM(patterns_analyzed) as week_analyses,
                        SUM(patterns_generated) as week_generations,
                        SUM(total_uploads) as week_uploads
                    FROM app_statistics 
                    WHERE stat_date >= date('now', '-7 days')
                ''')
                
                weekly_stats = cursor.fetchone()
                
                # Get daily statistics for the last week
                cursor.execute('''
                    SELECT * FROM app_statistics 
                    WHERE stat_date >= date('now', '-7 days')
                    ORDER BY stat_date DESC
                ''')
                
                daily_stats = [dict(row) for row in cursor.fetchall()]
                
                # Calculate classification accuracy (mock data for now)
                classification_accuracy = 0.89  # This would be calculated from actual results
                
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
                
                # Get or create today's record
                cursor.execute('''
                    INSERT OR IGNORE INTO app_statistics (stat_date) 
                    VALUES (date('now'))
                ''')
                
                # Update the specific statistic
                cursor.execute(f'''
                    UPDATE app_statistics 
                    SET {stat_name} = {stat_name} + ? 
                    WHERE stat_date = date('now')
                ''', (increment,))
                
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
                
                canvas_json = json.dumps(canvas_data, default=str)
                analysis_json = json.dumps(analysis_results, default=str) if analysis_results else None
                
                cursor.execute('''
                    INSERT INTO drawing_sessions 
                    (canvas_data, analysis_results, session_duration, is_completed)
                    VALUES (?, ?, ?, ?)
                ''', (canvas_json, analysis_json, session_duration, analysis_results is not None))
                
                session_id = cursor.lastrowid
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
                cursor = conn.cursor()
                
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
                    WHERE upload_timestamp < datetime('now', '-{} days')
                    AND id NOT IN (
                        SELECT item_id FROM user_favorites WHERE item_type = 'analysis'
                    )
                '''.format(days_to_keep))
                
                analysis_deleted = cursor.rowcount
                
                # Clean up old non-favorite generated patterns
                cursor.execute('''
                    DELETE FROM generated_patterns 
                    WHERE generation_timestamp < datetime('now', '-{} days')
                    AND is_favorite = FALSE
                    AND id NOT IN (
                        SELECT item_id FROM user_favorites WHERE item_type = 'generated'
                    )
                '''.format(days_to_keep))
                
                generation_deleted = cursor.rowcount
                
                # Clean up old drawing sessions
                cursor.execute('''
                    DELETE FROM drawing_sessions 
                    WHERE session_timestamp < datetime('now', '-{} days')
                    AND is_completed = FALSE
                '''.format(days_to_keep))
                
                sessions_deleted = cursor.rowcount
                
                # Clean up old statistics (keep last 90 days)
                cursor.execute('''
                    DELETE FROM app_statistics 
                    WHERE stat_date < date('now', '-90 days')
                ''')
                
                stats_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleanup completed: {analysis_deleted} analyses, {generation_deleted} generations, "
                           f"{sessions_deleted} sessions, {stats_deleted} stats deleted")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def export_data(self, output_format='json'):
        """
        Export all data for backup or analysis
        
        Args:
            output_format: 'json' or 'csv'
            
        Returns:
            Exported data
        """
        try:
            with self.get_connection() as conn:
                if output_format.lower() == 'csv':
                    # Export as CSV using pandas
                    analyses_df = pd.read_sql_query('SELECT * FROM analysis_results', conn)
                    generations_df = pd.read_sql_query('SELECT * FROM generated_patterns', conn)
                    favorites_df = pd.read_sql_query('SELECT * FROM user_favorites', conn)
                    
                    return {
                        'analyses': analyses_df.to_csv(index=False),
                        'generations': generations_df.to_csv(index=False),
                        'favorites': favorites_df.to_csv(index=False)
                    }
                
                else:  # JSON format
                    cursor = conn.cursor()
                    
                    # Get all data
                    cursor.execute('SELECT * FROM analysis_results')
                    analyses = [dict(row) for row in cursor.fetchall()]
                    
                    cursor.execute('SELECT * FROM generated_patterns')
                    generations = [dict(row) for row in cursor.fetchall()]
                    
                    cursor.execute('SELECT * FROM user_favorites')
                    favorites = [dict(row) for row in cursor.fetchall()]
                    
                    cursor.execute('SELECT * FROM app_statistics')
                    statistics = [dict(row) for row in cursor.fetchall()]
                    
                    return {
                        'export_timestamp': datetime.now().isoformat(),
                        'analyses': analyses,
                        'generations': generations,
                        'favorites': favorites,
                        'statistics': statistics
                    }
                    
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return None

# Global database instance
_db_instance = None

def get_database():
    """Get the global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = KolamDatabase()
    return _db_instance

def init_database():
    """Initialize the database with all required tables"""
    try:
        db = get_database()
        db.create_tables()
        logger.info("Database initialization completed")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

# Convenience functions for common operations
def save_analysis(filename, analysis_data, classification_result=None):
    """Convenience function to save analysis results"""
    db = get_database()
    return db.save_analysis_result(filename, analysis_data, classification_result)

def save_generation(parameters, pattern_data, generation_time):
    """Convenience function to save generated patterns"""
    db = get_database()
    return db.save_generated_pattern(parameters, pattern_data, generation_time)

def add_favorite(item_type, item_id, notes=""):
    """Convenience function to add favorites"""
    db = get_database()
    return db.add_to_favorites(item_type, item_id, notes)

def get_stats():
    """Convenience function to get application statistics"""
    db = get_database()
    return db.get_app_statistics()

def get_recent_activity(limit=10):
    """Convenience function to get recent activity"""
    db = get_database()
    return {
        'analyses': db.get_recent_analyses(limit),
        'generations': db.get_recent_generations(limit)
    }

