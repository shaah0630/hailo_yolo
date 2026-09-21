from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface, InputVStreamParams

class HailoInferenceEngine:
    """Encapsulates Hailo-8L backbone + Python head inference"""
    
    def __init__(self, hef_path: str):
        """Initialize the Hailo inference engine"""
        self.target_vdev = VDevice()
        # Load compiled HEF to device
        self.hef = HEF(hef_path)
        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target_vdev.configure(self.hef, configure_params)[0]
        # Create input virtual streams params
        input_vstreams_params = InputVStreamParams.make(self.network_group)

        print(f"✓ Hailo engine initialized: {hef_path}")        
