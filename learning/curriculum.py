
class Curriculum:
    def __init__(self, sessions=None, strategy="sorted"):

        self.sessions = sessions if sessions is not None else {}
        self.strategy = strategy
    
    def add_session(self, session_obj):
        """Adds a session to the curriculum."""
        self.sessions.append(session_obj)
    
    def get_sessions(self):
        if self.strategy == "sorted":
            return self.sessions
        else:
            pass # TODO
        
        return self.sessions