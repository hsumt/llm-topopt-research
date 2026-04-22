class PhysicsGate:
    def check(self, state):
        # 1. Is the structure still connected/functional/printable?
        # 2. Is there checkerboarding?
        # 3. Did the volume fraction overshoot the ceiling?
        pass
class SteeringAgent:
    # Was thinking this Agent could check every 10 or so iterations of SIMP to make sure it makes sense
    # and to occasionally adjust the penalization factor (maybe it was at 3 --> agent changes it to 3.4)
    # so change filter radius if there are thin hairs appearing
    # % grey. Never let it check the checkboarding, let the class and math do it
    def hello(idk):
        pass

class CriticAgent:
    def summarize(self, final_mesh_data):
        # I Imagine it saying something like:
        # "The final cantilever beam shows a clear truss topology. 
        #  However, the connection at the support is thin, suggesting 
        #  we should increase the volume fraction in the next run."
        pass