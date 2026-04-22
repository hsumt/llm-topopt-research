# simp v2

from agents import PhysicsGate, SteeringAgent


class SIMPSolver:
    def optimize(self):
        self.setup_fenicsx_physics()
        self.agent = SteeringAgent(config=self.config)
        self.gate = PhysicsGate(target_vol=self.volfrac)

        for itr in range(self.maxiter):
            # calc displacement
            #calc sensitivity
            # OC Update
            
            #snapshot

            if itr % 10 == 0:
                print(f"--- Agent Review ---")

    self.critic.generate_critique


# What's a good start p value? 3? Or should I start low and ramp to 3? Or should I steer based on the grey-fraction?
# Online it said do 40% for the volume fraction. (30-50%) why not less? Are most parts 0.4?
# 