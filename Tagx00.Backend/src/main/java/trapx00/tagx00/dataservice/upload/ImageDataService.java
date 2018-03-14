package trapx00.tagx00.dataservice.upload;

import trapx00.tagx00.exception.viewexception.SystemException;

public interface ImageDataService {
    /**
     * upload the image to the oos cloud
     *
     * @param key   the id of the image
     * @param bytes the image content
     * @return the url of the uploaded image
     */
    String uploadImage(String key, byte[] bytes) throws SystemException;

    /**
     * delete the image
     *
     * @param key the id of the image
     */
    void deleteImage(String key);
}
