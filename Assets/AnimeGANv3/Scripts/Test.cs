using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;

public class Test : MonoBehaviour
{
    public int webcamWidth = 640;
    public int webcamHeight = 480;
    private WebCamTexture webCamTexture;

    public ModelAsset modelAsset;
    private Worker _worker;
    private Model _runtimeModel;
    private Tensor<float> _inputTensor;

    public RawImage rawImage;

    private void Awake()
    {
        _runtimeModel = ModelLoader.Load(modelAsset);
        _worker = new Worker(_runtimeModel, BackendType.GPUCompute);
        _inputTensor = new Tensor<float>(new TensorShape(1, webcamWidth, webcamHeight, 3));

        InitializeWebcam();
    }

    void InitializeWebcam()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            webCamTexture = new WebCamTexture(devices[0].name, webcamWidth, webcamHeight, 30);
            webCamTexture.Play();
            Debug.Log($"启动摄像头: {devices[0].name}");
        }
        else
        {
            Debug.LogWarning("未找到摄像头设备");
        }
    } 

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        ProcessFrame(WebCamTextureToTexture2D(webCamTexture));
    }

    void ProcessFrame(Texture2D sourceTexture)
    {
        int textureWidth = sourceTexture.width;
        int textureHeight = sourceTexture.height;
        _inputTensor = TextureToTensor(sourceTexture);
        _worker.Schedule(_inputTensor);

        var outputTensor = _worker.PeekOutput() as Tensor<float>;

        var output = outputTensor.ReadbackAndClone();
         
        if (rawImage != null)
        {
            rawImage.texture = TensorToTexture(output);
        }
    }

    Texture2D WebCamTextureToTexture2D(WebCamTexture webCamTexture)
    {
        Texture2D tex = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGBA32, false);
        tex.SetPixels32(webCamTexture.GetPixels32());
        tex.Apply();
        return tex;
    } 

    private Tensor<float> TextureToTensor(Texture2D texture)
    {
        // 将Texture2D转换为Tensor<float>，形状为[1, height, width, 3]
        int width = texture.width;
        int height = texture.height;
        var inputShape = new TensorShape(1, height, width, 3);
        float[] data = new float[3 * height * width];
        var tensor = new Tensor<float>(inputShape, data);
        // 获取纹理的像素数据
        Color32[] pixels = texture.GetPixels32();

        // 将像素数据填充到张量中，注意颜色通道顺序和归一化
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                // 将颜色从Color32转换为float，并归一化到[-1,1]（根据原始代码的输入范围）
                // 注意：原始代码中输入是[-1,1]，所以这里需要将像素值从[0,255]转换为[-1,1]
                tensor[0, y, x, 0] = (pixels[index].r / 127.5f) - 1.0f;
                tensor[0, y, x, 1] = (pixels[index].g / 127.5f) - 1.0f;
                tensor[0, y, x, 2] = (pixels[index].b / 127.5f) - 1.0f;
            }
        }
        return tensor;
    }

    private Texture2D TensorToTexture(Tensor<float> tensor)
    {
        // 假设张量形状为[1, height, width, 3]
        int height = tensor.shape[1];
        int width = tensor.shape[2];

        Texture2D texture = new Texture2D(width, height);
        Color32[] pixels = new Color32[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // 将张量中的值从[-1,1]转换回[0,255]
                float r = (tensor[0, y, x, 0] + 1.0f) * 127.5f;
                float g = (tensor[0, y, x, 1] + 1.0f) * 127.5f;
                float b = (tensor[0, y, x, 2] + 1.0f) * 127.5f;

                pixels[y * width + x] = new Color32((byte)r, (byte)g, (byte)b, 255);
            }
        }

        texture.SetPixels32(pixels);
        texture.Apply();
        return texture;
    }

    void OnDestroy()
    {
        _inputTensor?.Dispose();
        _worker?.Dispose();
    }
}
