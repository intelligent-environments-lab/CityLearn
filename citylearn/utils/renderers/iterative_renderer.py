from renderer import Renderer
from datetime import date

class IterativeRenderer(Renderer):

    
    def __init__(self, directory: str, flag : str, session_name : str,
                start_date : date, enabled = True):
        
        super().__init__(directory, flag, session_name,
                        start_date, enabled)
        


    def export_csv(self, filename : str, data : dict):
        pass
