class InferenceEngineBase {
public:
    virtual preprocess() = 0;
    virtual forward() = 0;
    virtual postprocess() = 0;

    virtual ~InferenceEngineBase() = default;
};