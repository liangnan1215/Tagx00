package trapx00.tagx00.dataservice.upload;

import trapx00.tagx00.exception.viewexception.SystemException;
import trapx00.tagx00.exception.viewexception.TextNotExistException;

public interface TextDataService {
    /**
     * save the 3d
     *
     * @param token the token of the text
     * @param text  the content of the text
     * @return the token of the uploaded text
     */
    String uploadText(String token, String text) throws SystemException;

    /**
     * get 3d by its token
     *
     * @param token
     * @return
     */
    String getText(String token) throws TextNotExistException, SystemException;

    /**
     * delete the 3d
     *
     * @param token the token of the text
     */
    void deleteThreeDimension(String token);
}
